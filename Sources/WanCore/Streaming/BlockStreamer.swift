// HV2 weight streaming, part 2: the runtime (NEUROSTREAM-ACTIONS HV2).
// Streams a WanModel's transformer blocks from per-block granule files (see
// GranuleLayout.swift) through TWO resident block-group slots, refilled by a
// background pread thread while the GPU computes the other slot — the pread
// double-buffer proven bit-exact at scale by spikes/hv2_stream_proto
// (probes/hv2_stream_proto.out) over the P-C alias seam (probes/pc_mlx_seam.out).
//
// Mechanics carried verbatim from the prototype (do not re-derive):
//   - Slot-backed parameter arrays are injected into the blocks ONCE at bind;
//     refills mutate the backings through `asMTLBuffer(device:noCopy:true)`
//     (public mlx-swift API; pointer stability re-verified at bind).
//   - STEP-MAJOR loop order: the cyclic group sequence advances continuously
//     across sweeps (KV precompute, then every denoise forward) — never
//     block-major (teardown §2.3: 64× less IO).
//   - Compute `eval`s at every group boundary BEFORE releasing the slot — the
//     handoff that makes refills invisible to the lazy graph (and the reason
//     streamed output is bit-identical to resident).
//   - IO runs on a plain nonisolated background thread and never touches MLX;
//     compute stays on the caller's isolation (@InferenceActor in the engine —
//     the specialty-surface split).
//   - Granules are read F_NOCACHE: bounded memory behavior is the point of
//     streaming, so the page cache must not silently absorb the model (and S
//     measurements stay honest — the prototype's cold-number recipe).
//
// Self-calibrating runtime gate (prototype phase G, all six sweep points):
//   stream hides iff S ≥ C(N)  ⇔  N ≥ B·F/(2·S)
// with S measured on the first refill sweep (the streamed KV precompute) and
// F/C measured IN-REGIME during the first streamed forward (sustained-load
// thermal drift halves large-N throughput within minutes — receipt item 6 —
// so a cold synthetic calibration would overstate the threshold). Policy
// `.auto` falls back to fully-resident automatically (loading the blocks from
// the granules) when the arithmetic doesn't clear — a big-GPU machine at
// image-class N is exactly the corner that must fall back; video-class N
// (33k–76k tokens) clears everywhere measured.

import Foundation
import MLX
import MLXNN
import Metal
import os

public enum BlockStreamerError: LocalizedError {
    case noMetalDevice
    case aliasUnavailable(String)
    case pointerInstability(String)
    case contract(String)
    case state(String)

    public var errorDescription: String? {
        switch self {
        case .noMetalDevice: return "BlockStreamer: no Metal device"
        case .aliasUnavailable(let m):
            return "BlockStreamer: asMTLBuffer(noCopy:true) unavailable for \(m) — seam broken"
        case .pointerInstability(let m):
            return "BlockStreamer: slot backing pointer moved (\(m)) — refusing to stream"
        case .contract(let m): return "BlockStreamer contract violation: \(m)"
        case .state(let m): return "BlockStreamer state error: \(m)"
        }
    }
}

/// Safetensors dtype string → MLX DType (the subset Wan checkpoints use).
func mlxDType(safetensors dtype: String) throws -> DType {
    switch dtype {
    case "BF16": return .bfloat16
    case "F16": return .float16
    case "F32": return .float32
    case "U32": return .uint32
    case "U8": return .uint8
    case "I32": return .int32
    case "I64": return .int64
    default:
        throw BlockStreamerError.contract("unsupported safetensors dtype \(dtype)")
    }
}

/// The raw backing pointer of an evaluated, contiguous MLXArray — the P-C alias
/// seam. `asData(access: .noCopyIfContiguous)` wraps the SAME pointer
/// (`mlx_array_data_uint8`) that `asMTLBuffer(noCopy: true)` exposes, but works
/// for tensors below page alignment (norm weights are 10 KiB; `makeBuffer(
/// bytesNoCopy:)` needs page-aligned pointers and page-multiple lengths). A
/// silent copy (non-contiguous backing) is detected by double extraction: a
/// wrapper returns the same address twice, a copy cannot.
func backingPointer(_ a: MLXArray, context: @autoclosure () -> String) throws
    -> UnsafeMutableRawPointer
{
    func extract() -> UnsafeMutableRawPointer? {
        let d = a.asData(access: .noCopyIfContiguous)
        return d.data.withUnsafeBytes { raw in
            raw.baseAddress.map { UnsafeMutableRawPointer(mutating: $0) }
        }
    }
    guard let p1 = extract(), let p2 = extract(), p1 == p2 else {
        throw BlockStreamerError.aliasUnavailable(context())
    }
    return p1
}

/// Opt-in configuration for block streaming. Constructed by the consumer's
/// package configuration (per config, never ambient) and handed to
/// `BlockStreamer.init` alongside the granule directories.
public struct BlockStreamingOptions: Sendable {
    /// Blocks per granule group (slot capacity). Must divide the block count.
    /// 2 (the prototype's shape) keeps slots ~1.4 GB bf16 / ~0.4 GB int4 on A14B.
    public var groupSize: Int
    /// `.auto` — evaluate the runtime gate after the first streamed forward and
    /// fall back to fully-resident (silently, one log line) when it doesn't
    /// clear. `.forceStream` — never fall back (receipts, parity tests).
    public var gatePolicy: GatePolicy
    /// Extra safety margin on the gate: require S ≥ C·margin. 1.0 = the measured
    /// prototype constant (stall tracked the un-margined gate at all six points).
    public var gateMargin: Double
    /// Suppress the one-line lifecycle prints.
    public var quiet: Bool

    public enum GatePolicy: String, Sendable {
        case auto
        case forceStream
    }

    public init(
        groupSize: Int = 2,
        gatePolicy: GatePolicy = .auto,
        gateMargin: Double = 1.0,
        quiet: Bool = false
    ) {
        self.groupSize = groupSize
        self.gatePolicy = gatePolicy
        self.gateMargin = gateMargin
        self.quiet = quiet
    }
}

/// What the gate measured and decided — surfaced for receipts and logs.
public struct BlockStreamerGateReport: Sendable {
    public var n: Int  // tokens in the gating forward (batch × seqLen)
    public var sGiBs: Double  // measured refill bandwidth
    public var cGiBs: Double  // measured GPU weight-consumption rate C(N)
    public var nMin: Int  // N_min = N · C / S at the measured rates
    public var step1ComputeSeconds: Double
    public var step1StallSeconds: Double
    public var streaming: Bool  // the decision
}

public final class BlockStreamer: @unchecked Sendable {

    public enum Verdict: String, Sendable {
        case undecided  // gate not yet evaluated (before the first streamed forward)
        case streaming  // gate cleared (or .forceStream) — streaming stays on
        case fellBack  // gate failed under .auto — blocks were made resident
    }

    // Immutable layout after init.
    let options: BlockStreamingOptions
    let granuleDirs: [URL]
    let manifests: [GranuleManifest]
    public let blockCount: Int
    public let groupSize: Int
    public let numGroups: Int
    let template: [GranuleTensor]  // uniform per-block tensor table
    let tensorsPerBlock: Int
    /// Raw streamed bytes per full sweep (one expert's blocks).
    public let sweepBytes: Int
    /// Resident slot footprint: 2 slots × groupSize blocks of parameters.
    public let slotResidentBytes: Int

    private let device: any MTLDevice
    private let lock = OSAllocatedUnfairLock()

    // Slot storage — allocated at bind, stable for the streamer's lifetime.
    // Flat layout: [slot][localBlock * tensorsPerBlock + t]. The arrays OWN the
    // backing memory; slotPtrs alias it (valid while the arrays live).
    private var slotArrays: [[MLXArray]] = []
    private var slotPtrs: [[UnsafeMutableRawPointer]] = []

    private var experts: [WanModel] = []
    private var expertIndex: [ObjectIdentifier: Int] = [:]

    // IO machinery — one instance per activation (per expert granule set). The
    // refill plan (raw slot pointers + granule offsets) lives INSIDE this
    // @unchecked Sendable box so the @Sendable Thread body captures only it
    // (the prototype's IOBox shape; pointers cross the thread behind the
    // free/ready semaphore protocol).
    private final class IOState: @unchecked Sendable {
        struct Refill {
            let fd: Int32  // granule file of the block
            let dst: UnsafeMutableRawPointer  // slot tensor backing
            let offset: Int
            let nbytes: Int
        }

        let fds: [Int32]
        /// [slot][flat refill index] — every pread needed to fill group g into
        /// a slot is `plan[slot][g]`.
        let plan: [[[Refill]]]  // [slot][group][refill]
        let free: [DispatchSemaphore]
        let ready: [DispatchSemaphore]
        let done = DispatchSemaphore(value: 0)
        let lock = OSAllocatedUnfairLock()
        var cancelled = false
        var ioBusySeconds: Double = 0
        var ioBytes: Int = 0
        var failure: String? = nil

        init(fds: [Int32], plan: [[[Refill]]]) {
            self.fds = fds
            self.plan = plan
            // Created at 0 and primed by signal() so dealloc is legal at any
            // rest value (a DispatchSemaphore whose value sits below its
            // creation value at dealloc traps in libdispatch).
            self.free = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
            self.ready = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
            free[0].signal()
            free[1].signal()
        }

        func note(busy: Double, bytes: Int) {
            lock.lock()
            ioBusySeconds += busy
            ioBytes += bytes
            lock.unlock()
        }

        var isCancelled: Bool {
            lock.lock()
            defer { lock.unlock() }
            return cancelled
        }

        func cancel() {
            lock.lock()
            cancelled = true
            lock.unlock()
        }
    }

    private var io: IOState?
    private var activeExpert = -1
    private var computeSeq = 0  // compute-side cyclic sequence (compute thread only)

    // Gate state (compute thread only, except `verdict` reads).
    private var _verdict: Verdict = .undecided
    private var _gateReport: BlockStreamerGateReport? = nil
    private var forwardCompute: Double = 0
    private var forwardStall: Double = 0
    private var forwardTokens: Int = 0
    private var groupOpenedAt: Double = 0

    public var verdict: Verdict {
        lock.lock()
        defer { lock.unlock() }
        return _verdict
    }

    public var gateReport: BlockStreamerGateReport? {
        lock.lock()
        defer { lock.unlock() }
        return _gateReport
    }

    /// Cumulative refill bandwidth over the current activation (the measured S).
    public var measuredSGiBs: Double {
        guard let io else { return 0 }
        io.lock.lock()
        defer { io.lock.unlock() }
        return io.ioBusySeconds > 0
            ? Double(io.ioBytes) / io.ioBusySeconds / 1_073_741_824 : 0
    }

    // MARK: - Init / bind

    /// One granule directory per expert (dense models pass one; A14B passes
    /// [high, low]). All manifests must describe identical block layouts.
    public init(granuleDirs: [URL], options: BlockStreamingOptions = .init()) throws {
        guard !granuleDirs.isEmpty else {
            throw BlockStreamerError.contract("no granule directories")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BlockStreamerError.noMetalDevice
        }
        self.device = device
        self.options = options
        self.granuleDirs = granuleDirs
        self.manifests = try granuleDirs.map { try GranuleManifest.load(from: $0) }

        let first = manifests[0]
        for m in manifests.dropFirst() {
            let a = first.blocks[0].tensors.map { "\($0.key)|\($0.dtype)|\($0.shape)" }
            let b = m.blocks[0].tensors.map { "\($0.key)|\($0.dtype)|\($0.shape)" }
            guard m.blockCount == first.blockCount, a == b else {
                throw BlockStreamerError.contract(
                    "expert granule sets differ in layout — cannot share slots")
            }
        }
        self.blockCount = first.blockCount
        self.groupSize = options.groupSize
        guard groupSize > 0, blockCount % groupSize == 0 else {
            throw BlockStreamerError.contract(
                "groupSize \(groupSize) must divide blockCount \(blockCount)")
        }
        self.numGroups = blockCount / groupSize
        self.template = first.blocks[0].tensors
        self.tensorsPerBlock = template.count
        self.sweepBytes = first.streamedBytes
        self.slotResidentBytes = 2 * groupSize * template.reduce(0) { $0 + $1.nbytes }
    }

    public func blockRange(_ group: Int) -> Range<Int> {
        (group * groupSize)..<((group + 1) * groupSize)
    }

    /// Allocate the two slots and inject the slot-backed parameter arrays into
    /// every bound expert's blocks (once — the P-C seam). Experts must be
    /// freshly constructed (and, for quantized checkpoints, already passed
    /// through `WeightLoader.applyQuantization`) with their block parameters
    /// UNLOADED; global tensors load separately via `loadStreamingGlobals`.
    public func bind(experts boundExperts: [WanModel]) throws {
        guard boundExperts.count == manifests.count else {
            throw BlockStreamerError.contract(
                "\(boundExperts.count) experts vs \(manifests.count) granule sets")
        }
        guard slotArrays.isEmpty else {
            throw BlockStreamerError.state("bind() called twice")
        }

        // Contract check: the model's per-block parameter set must equal the
        // manifest's tensor table — 0 missing / 0 unused, shapes equal (the
        // refuse-partial-loads doctrine, streamed edition). Dtypes come from
        // the manifest (construction dtype differs pre-load, exactly as with
        // `update(parameters:)` on a resident load).
        for expert in boundExperts {
            guard expert.blocks.count == blockCount else {
                throw BlockStreamerError.contract(
                    "model has \(expert.blocks.count) blocks, manifest \(blockCount)")
            }
            let params = Dictionary(
                uniqueKeysWithValues: expert.blocks[0].parameters().flattened())
            let modelKeys = Set(params.keys)
            let manifestKeys = Set(template.map(\.key))
            let missing = manifestKeys.subtracting(modelKeys)
            let unused = modelKeys.subtracting(manifestKeys)
            guard missing.isEmpty, unused.isEmpty else {
                throw BlockStreamerError.contract(
                    "block param set mismatch: \(missing.count) missing "
                        + "(e.g. \(missing.sorted().prefix(3))), \(unused.count) unused "
                        + "(e.g. \(unused.sorted().prefix(3)))")
            }
            for t in template {
                guard params[t.key]!.shape == t.shape else {
                    throw BlockStreamerError.contract(
                        "shape mismatch for \(t.key): model \(params[t.key]!.shape) "
                            + "vs manifest \(t.shape)")
                }
            }
        }

        // Allocate slot arrays (zeros, eval'd, aliased — the prototype recipe).
        // Where page alignment permits, cross-check the pointer against the
        // exact P-C-proven `asMTLBuffer(noCopy: true)` route.
        for _ in 0..<2 {
            var arrays: [MLXArray] = []
            var ptrs: [UnsafeMutableRawPointer] = []
            for _ in 0..<groupSize {
                for t in template {
                    let a = MLXArray.zeros(t.shape, dtype: try mlxDType(safetensors: t.dtype))
                    eval(a)
                    let ptr = try backingPointer(a, context: "slot \(t.key)")
                    if UInt(bitPattern: ptr) % 16384 == 0, t.nbytes % 16384 == 0,
                        let buf = a.asMTLBuffer(device: device, noCopy: true),
                        buf.contents() != ptr
                    {
                        throw BlockStreamerError.pointerInstability(
                            "asData vs asMTLBuffer disagree for \(t.key)")
                    }
                    arrays.append(a)
                    ptrs.append(ptr)
                }
            }
            slotArrays.append(arrays)
            slotPtrs.append(ptrs)
        }

        // Inject: block i of every expert points at slot (i/G)%2, local i%G.
        for expert in boundExperts {
            for (i, block) in expert.blocks.enumerated() {
                let slot = (i / groupSize) % 2
                let base = (i % groupSize) * tensorsPerBlock
                let pairs = template.enumerated().map { (t, entry) in
                    (entry.key, slotArrays[slot][base + t])
                }
                try block.update(
                    parameters: ModuleParameters.unflattened(pairs),
                    verify: [.noUnusedKeys])
            }
        }

        // Pointer stability re-verified at bind (the plan's requirement): every
        // injected parameter must alias the exact recorded slot pointer.
        for expert in boundExperts {
            for (i, block) in expert.blocks.enumerated() {
                let slot = (i / groupSize) % 2
                let base = (i % groupSize) * tensorsPerBlock
                let params = Dictionary(
                    uniqueKeysWithValues: block.parameters().flattened())
                for (t, entry) in template.enumerated() {
                    let ptr = try backingPointer(
                        params[entry.key]!, context: "\(entry.key) post-inject")
                    guard ptr == slotPtrs[slot][base + t] else {
                        throw BlockStreamerError.pointerInstability(
                            "block \(i) \(entry.key)")
                    }
                }
            }
        }

        experts = boundExperts
        expertIndex = Dictionary(
            uniqueKeysWithValues: boundExperts.enumerated().map {
                (ObjectIdentifier($1), $0)
            })
        for expert in boundExperts {
            expert.blockStreamer = self
        }
        lock.lock()
        _verdict = options.gatePolicy == .forceStream ? .streaming : .undecided
        lock.unlock()
        say(String(
            format: "bound %d expert(s) · %d blocks · group=%d · slots 2×%.0f MiB "
                + "(%.2f GiB resident) · sweep %.2f GiB",
            experts.count, blockCount, groupSize,
            Double(slotResidentBytes) / 2 / 1_048_576,
            Double(slotResidentBytes) / 1_073_741_824,
            Double(sweepBytes) / 1_073_741_824))
    }

    /// Load the NON-block ("global") tensors — embeddings, head, … — from the
    /// source safetensors into a bound expert. Loads exactly the model's
    /// non-block parameter set; manifest-listed extras the model never loads
    /// (the int4 checkpoints' serialized `freqs`) are tolerated and skipped.
    public func loadStreamingGlobals(expert: WanModel, from safetensors: URL) throws {
        guard let index = expertIndex[ObjectIdentifier(expert)] else {
            throw BlockStreamerError.state("expert not bound")
        }
        let blockPrefix = manifests[index].blockPrefix
        let wanted = Set(
            expert.parameters().flattened().map(\.0).filter { !$0.hasPrefix(blockPrefix) })
        let available = Set(manifests[index].globalKeys)
        let missing = wanted.subtracting(available)
        guard missing.isEmpty else {
            throw BlockStreamerError.contract(
                "globals missing from checkpoint: \(missing.sorted().prefix(5))")
        }
        let loaded = try Device.withDefaultDevice(.cpu) {
            let all = try MLX.loadArrays(url: safetensors)
            let picked = all.filter { wanted.contains($0.key) }
            WeightLoader.materialize(picked)
            return picked
        }
        try expert.update(
            parameters: ModuleParameters.unflattened(Array(loaded)),
            verify: [.noUnusedKeys])
        eval(expert.parameters())  // globals only touch resident arrays; slots are zeros
    }

    // MARK: - Activation (per-expert granule set) and the IO thread

    /// Make `model`'s granule set the streamed one, (re)starting the prefetch
    /// thread at group 0. Called automatically by the streamed forward/KV paths;
    /// switching experts (the A14B t=875 boundary) is just a different `model`
    /// arriving here.
    func ensureActive(for model: WanModel) {
        guard let index = expertIndex[ObjectIdentifier(model)] else {
            fatalError("BlockStreamer: forward on an unbound expert")
        }
        if index == activeExpert, io != nil { return }
        stopIO()
        startIO(expert: index)
    }

    private func startIO(expert: Int) {
        precondition(io == nil)
        let dir = granuleDirs[expert]
        let manifest = manifests[expert]
        let fds: [Int32] = manifest.blocks.map { block in
            let path = dir.appendingPathComponent(block.file).path
            guard let fd = try? GranuleIO.openRead(path) else {
                fatalError("BlockStreamer: cannot open granule \(path)")
            }
            return fd
        }
        // Precompute the full refill plan: [slot][group] → the preads that fill
        // that group's blocks into that slot's tensor backings.
        var plan: [[[IOState.Refill]]] = []
        for slot in 0..<2 {
            var groups: [[IOState.Refill]] = []
            for g in 0..<numGroups {
                var refills: [IOState.Refill] = []
                for lb in 0..<groupSize {
                    let bi = g * groupSize + lb
                    for (t, entry) in manifest.blocks[bi].tensors.enumerated() {
                        refills.append(
                            IOState.Refill(
                                fd: fds[bi],
                                dst: slotPtrs[slot][lb * tensorsPerBlock + t],
                                offset: entry.offset,
                                nbytes: entry.nbytes))
                    }
                }
                groups.append(refills)
            }
            plan.append(groups)
        }
        let state = IOState(fds: fds, plan: plan)
        io = state
        activeExpert = expert
        computeSeq = 0

        let numGroups = self.numGroups
        let thread = Thread {
            var seq = 0
            while !state.isCancelled {
                let slot = seq % 2
                state.free[slot].wait()
                if state.isCancelled { break }
                let g = seq % numGroups
                let t0 = CFAbsoluteTimeGetCurrent()
                var bytes = 0
                for refill in state.plan[slot][g] {
                    do {
                        try GranuleIO.preadFull(
                            fd: refill.fd, into: refill.dst,
                            count: refill.nbytes, offset: refill.offset)
                    } catch {
                        state.lock.lock()
                        state.failure = "\(error)"
                        state.cancelled = true
                        state.lock.unlock()
                        // Signal ready so a waiting compute thread can observe
                        // the failure instead of deadlocking.
                        state.ready[slot].signal()
                        state.done.signal()
                        return
                    }
                    bytes += refill.nbytes
                }
                state.note(busy: CFAbsoluteTimeGetCurrent() - t0, bytes: bytes)
                state.ready[slot].signal()
                seq += 1
            }
            state.done.signal()
        }
        thread.name = "wan-core.BlockStreamer.io"
        thread.qualityOfService = .userInitiated
        thread.start()
        say("streaming expert \(expert) from \(dir.lastPathComponent)")
    }

    private func stopIO() {
        guard let state = io else { return }
        state.cancel()
        // Unblock a thread parked on either free semaphore, then wait it out.
        state.free[0].signal()
        state.free[1].signal()
        state.done.wait()
        for fd in state.fds { close(fd) }
        io = nil
        activeExpert = -1
    }

    /// Stop the prefetch thread and close granule files. Slots and bindings
    /// stay — a later forward reactivates. Call at the end of a generation
    /// (or before releasing the streamer).
    public func finish() {
        stopIO()
    }

    deinit {
        stopIO()
    }

    // MARK: - Group window (compute side)

    /// Wait until the current group's slot is refilled. Returns the slot index.
    @discardableResult
    func acquireGroup() -> Int {
        guard let state = io else {
            fatalError("BlockStreamer: acquireGroup with no active IO")
        }
        let slot = computeSeq % 2
        let t0 = CFAbsoluteTimeGetCurrent()
        state.ready[slot].wait()
        if let failure = state.failure {
            fatalError("BlockStreamer IO failed: \(failure)")
        }
        if let poison = pendingPoison, poison.target == acquireCount {
            // Negative-control hook: corrupt the slot AFTER its prefetch
            // completed and BEFORE compute reads it (the prototype's control).
            let entry = template[poison.tensor]
            memset(
                slotPtrs[slot][poison.localBlock * tensorsPerBlock + poison.tensor],
                0x55, min(poison.bytes, entry.nbytes))
            pendingPoison = nil
            say("poisoned slot \(slot) at acquire #\(acquireCount) (negative control)")
        }
        acquireCount += 1
        let now = CFAbsoluteTimeGetCurrent()
        forwardStall += now - t0
        groupOpenedAt = now
        return slot
    }

    /// Hand the group's slot back to the IO thread. The caller MUST have
    /// `eval`'d everything that reads the slot's arrays before calling this.
    func releaseGroup() {
        guard let state = io else { return }
        let slot = computeSeq % 2
        forwardCompute += CFAbsoluteTimeGetCurrent() - groupOpenedAt
        state.free[slot].signal()
        computeSeq += 1
    }

    // MARK: - Gate

    func beginForward(tokens: Int) {
        forwardCompute = 0
        forwardStall = 0
        forwardTokens = tokens
    }

    /// Per-forward wrap-up: on the FIRST streamed forward, evaluate the runtime
    /// gate from in-regime measurements and (policy `.auto`) fall back to fully
    /// resident when it doesn't clear. Safe to call every forward — later calls
    /// only refresh the last-forward stats.
    func endForward() {
        lastForwardComputeSeconds = forwardCompute
        lastForwardStallSeconds = forwardStall
        guard verdict == .undecided else { return }

        let s = measuredSGiBs
        let c = forwardCompute > 0
            ? Double(sweepBytes) / forwardCompute / 1_073_741_824 : .infinity
        let nMin = s > 0 ? Int((Double(forwardTokens) * c / s).rounded(.up)) : Int.max
        let clears = s >= c * options.gateMargin
        let report = BlockStreamerGateReport(
            n: forwardTokens, sGiBs: s, cGiBs: c, nMin: nMin,
            step1ComputeSeconds: forwardCompute,
            step1StallSeconds: forwardStall,
            streaming: clears)
        say(String(
            format: "gate: N=%d · S=%.2f GiB/s vs C(N)=%.2f GiB/s → N_min≈%d → %@",
            report.n, s, c, nMin,
            clears ? "STREAM (hidden)" : "FALL BACK (resident)"))

        lock.lock()
        _gateReport = report
        _verdict = clears ? .streaming : .fellBack
        lock.unlock()

        if !clears {
            do {
                try fallBackResident()
            } catch {
                // Fallback is a strictly-safer path; failing it is fatal-worthy,
                // but streaming remains correct — keep streaming and say so.
                say("fallback FAILED (\(error)) — continuing streamed")
                lock.lock()
                _verdict = .streaming
                lock.unlock()
            }
        }
    }

    public private(set) var lastForwardComputeSeconds: Double = 0
    public private(set) var lastForwardStallSeconds: Double = 0

    // MARK: - Fully-resident fallback

    /// Load every bound expert's blocks resident from the granule files and
    /// detach the streamer. The already-computed streamed artifacts (KV caches,
    /// step-1 latents) are bit-exact and remain valid.
    public func fallBackResident() throws {
        stopIO()
        let t0 = CFAbsoluteTimeGetCurrent()
        var loadedBytes = 0
        for (e, expert) in experts.enumerated() {
            let dir = granuleDirs[e]
            let manifest = manifests[e]
            for (i, block) in expert.blocks.enumerated() {
                let granule = manifest.blocks[i]
                let fd = try GranuleIO.openRead(
                    dir.appendingPathComponent(granule.file).path)
                defer { close(fd) }
                var pairs: [(String, MLXArray)] = []
                var arrays: [MLXArray] = []
                for t in granule.tensors {
                    let a = MLXArray.zeros(t.shape, dtype: try mlxDType(safetensors: t.dtype))
                    eval(a)
                    let ptr = try backingPointer(a, context: "fallback \(t.key)")
                    try GranuleIO.preadFull(
                        fd: fd, into: ptr, count: t.nbytes, offset: t.offset)
                    pairs.append((t.key, a))
                    arrays.append(a)
                    loadedBytes += t.nbytes
                }
                try block.update(
                    parameters: ModuleParameters.unflattened(pairs),
                    verify: [.noUnusedKeys])
                eval(arrays)
            }
            expert.blockStreamer = nil
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        say(String(
            format: "fell back resident: %.2f GiB in %.1fs (%.2f GiB/s)",
            Double(loadedBytes) / 1_073_741_824, dt,
            Double(loadedBytes) / dt / 1_073_741_824))
        lock.lock()
        _verdict = .fellBack
        lock.unlock()
    }

    // MARK: - Test / receipt hooks

    private struct PendingPoison {
        let target: Int  // absolute acquire index (never resets)
        let localBlock: Int
        let tensor: Int
        let bytes: Int
    }
    private var pendingPoison: PendingPoison? = nil
    /// Monotonic count of group acquisitions across the streamer's lifetime
    /// (unlike computeSeq it never resets on expert switches) — the poison
    /// hook's clock. Compute-thread only.
    private var acquireCount = 0

    /// Arm a one-shot poisoned-slot negative control (a parity compare that
    /// cannot see a bad refill is not evidence): on the `afterAcquires`-th
    /// group acquisition from now (0 = the very next one), `bytes` of the given
    /// tensor are corrupted after the prefetch and before compute reads them.
    /// Counted across activations, so callers can target a specific sweep:
    /// e.g. with KV sweeps done, `3 * numGroups + 3` lands inside denoise
    /// step 2 (after step 1's re-activation sweep and one full step sweep).
    public func armPoison(
        afterAcquires: Int, localBlock: Int = 0, tensor: Int = 0, bytes: Int = 1 << 20
    ) {
        pendingPoison = PendingPoison(
            target: acquireCount + afterAcquires, localBlock: localBlock,
            tensor: tensor, bytes: bytes)
    }

    /// Stop IO and detach from every bound expert WITHOUT loading anything
    /// resident — the receipts' between-arms reset. Block parameters keep
    /// aliasing this streamer's slots until another `bind` replaces them, so
    /// no forward may run between `detach()` and the next bind.
    public func detach() {
        stopIO()
        for expert in experts {
            expert.blockStreamer = nil
        }
    }

    private func say(_ message: String) {
        if !options.quiet {
            print("[BlockStreamer] \(message)")
            fflush(stdout)
        }
    }
}
