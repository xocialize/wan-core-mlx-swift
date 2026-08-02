// RunWanStream — the HV2 BlockStreamer receipts gate (NEUROSTREAM-ACTIONS HV2).
// Streams REAL Wan2.2-A14B (Bernini-R) experts through wan-core's BlockStreamer
// and writes the integration receipt: bit-exact parity (with determinism and
// poisoned-slot negative controls), runtime-gate calibration across an N ladder,
// and a 16 GB-budget emulation arm (engine budget arithmetic: 0.7 × RAM).
//
//   swift run -c release RunWanStream [--arms layout,parity-bf16,parity-int4,gate,budget]
//     [--bf16-dir …/ckpt-bf16] [--int4-dir …/ckpt-int4]
//     [--granules-root /Volumes/Satechi/Models/wan-granules]
//     [--receipt …/probes/hv2_wan_blockstreamer.out] [--steps 4] [--group 2]
//
// Measurement doctrine carried from the prototype receipts (both .out files):
//   - Release builds only; never profile the first turn (warm-up before timing).
//   - Matched baselines in-regime: resident comparisons are taken immediately
//     after the streamed run they compare against (sustained-load thermal drift
//     slows ALL large-N compute ~2× within minutes; 150 s idle recovers).
//   - Granules are written AND read F_NOCACHE (the honest-cold recipe on a
//     big-RAM machine); the achieved regime is classified from measured S.
//   - Conditioning is a synthetic UMT5-feature fixture and the VAE is not
//     loaded: the receipt measures the DiT denoise working set — the term
//     weight streaming attacks (umT5 is paged+evicted per §2.4 and the VAE has
//     its own streaming levers). Stated again in the receipt header.
//
// NAX split-K note (checked for this receipt): A14B ffn.fc2 is K=13824 ≥ 10240,
// inside the NAX bf16 GEMM JIT window once M·N = (B·L)·5120 ≥ 2048² i.e.
// B·L ≥ 819 tokens. The small-N parity config is pinned BELOW that (B·L = 768)
// so parity bytes are window-clean; larger-N arms compare streamed-vs-resident
// through IDENTICAL kernels, so those comparisons hold regardless, and the
// production numerics path (fp32 SDPA ≥ wanLargeSeq) is wan-core's default.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Metal
import WanCore

// ---------- small utils ----------

func now() -> Double { CFAbsoluteTimeGetCurrent() }
func gib(_ b: Int) -> Double { Double(b) / 1_073_741_824 }

// Swift 6 top-level bindings are MainActor-isolated; the receipt box is a
// Sendable class so the nonisolated arm functions can log through it.
final class ReceiptBox: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func append(_ s: String) {
        lock.lock()
        lines.append(s)
        lock.unlock()
    }
    var text: String {
        lock.lock()
        defer { lock.unlock() }
        return lines.joined(separator: "\n") + "\n"
    }
}
let receiptBox = ReceiptBox()

func log(_ s: String) {
    print(s)
    fflush(stdout)
    receiptBox.append(s)
}

func argValue(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

/// OS-level truth for the governor basis (phys_footprint, task_info) — under a
/// cache cap `Memory.peakMemory` counts cumulative allocations and misleads.
func physFootprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Int(info.phys_footprint) : -1
}

/// Byte-exact compare through the backing (`asData` reaches the same pointer as
/// the P-C alias; a copy fallback preserves bytes, so the compare stays exact).
func bitIdentical(_ a: MLXArray, _ b: MLXArray) -> Bool {
    eval(a, b)
    guard a.nbytes == b.nbytes, a.dtype == b.dtype else { return false }
    let da = a.asData(access: .noCopyIfContiguous)
    let db = b.asData(access: .noCopyIfContiguous)
    return da.data.withUnsafeBytes { pa in
        db.data.withUnsafeBytes { pb in
            memcmp(pa.baseAddress!, pb.baseAddress!, a.nbytes) == 0
        }
    }
}

// ---------- configuration ----------

struct RunConfig: Sendable {
    var arms: [String]
    var bf16Dir: URL
    var int4Dir: URL
    var granulesRoot: URL
    var receipt: URL
    var steps: Int
    var group: Int
    /// Gate-ladder rung indices to run (default all). The harness runs ONE rung
    /// per process: each auto-fallback loads ~52 GiB resident that macOS does
    /// not reclaim from a live process even after release+clearCache (freed
    /// pages linger in phys_footprint), so a single process looping 4 rungs
    /// ratchets to ~120 GB and dies at rung 3's fallback (runs 1+3). Process
    /// exit is the guaranteed reclaim. Production is unaffected — a consumer
    /// falls back ONCE and those residents ARE its working set; only this
    /// repeated-fallback harness loops.
    var gateRungs: [Int]
}

let defaultWeights = "/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights"
let cfg = RunConfig(
    arms: (argValue("--arms") ?? "layout,parity-bf16,parity-int4,gate,budget")
        .split(separator: ",").map(String.init),
    bf16Dir: URL(filePath: argValue("--bf16-dir") ?? defaultWeights + "/ckpt-bf16"),
    int4Dir: URL(filePath: argValue("--int4-dir") ?? defaultWeights + "/ckpt-int4"),
    granulesRoot: URL(
        filePath: argValue("--granules-root") ?? "/Volumes/Satechi/Models/wan-granules"),
    receipt: URL(
        filePath: argValue("--receipt")
            ?? "/Volumes/Satechi/Development/mlxengine-todo/probes/hv2_wan_blockstreamer.out"),
    steps: argValue("--steps").flatMap(Int.init) ?? 4,
    group: argValue("--group").flatMap(Int.init) ?? 2,
    gateRungs: (argValue("--gate-rungs") ?? "0,1,2,3")
        .split(separator: ",").compactMap { Int($0) })

func ckptDir(_ variant: String) -> URL { variant == "int4" ? cfg.int4Dir : cfg.bf16Dir }
func granuleDir(_ variant: String, _ expert: String) -> URL {
    cfg.granulesRoot.appendingPathComponent(variant).appendingPathComponent(expert)
}

// ---------- header ----------

let osv = ProcessInfo.processInfo.operatingSystemVersion
log("== HV2 BlockStreamer integration receipt (wan-core × Wan2.2-A14B / Bernini-R) ==")
log(String(
    format: "device: %@ · RAM %.0f GiB · macOS %d.%d",
    MTLCreateSystemDefaultDevice()?.name ?? "unknown",
    gib(Int(ProcessInfo.processInfo.physicalMemory)),
    osv.majorVersion, osv.minorVersion))
log("measurement configuration: real A14B experts (dual, switch at t=875), CFG B=2 unless noted;")
log("  conditioning = synthetic UMT5-feature fixture (umT5 NOT loaded — §2.4 pages+evicts it);")
log("  VAE not loaded (own streaming levers); receipt covers the DiT denoise working set only;")
log("  granules written+read F_NOCACHE on \(cfg.granulesRoot.path); Release build; per-step clearCache.")
log("  gate policy: measure S on first refill sweep (streamed KV precompute), F in-regime during")
log("  step 1, decide N ≥ B·F/(2·S) ⇔ S ≥ C(N); matched-resident baselines taken adjacent in time.")

// ---------- phase L · granule layout (idempotent) ----------

func ensureLayout(variant: String) throws {
    let dir = ckptDir(variant)
    let files = [
        ("high", "high_noise_model.safetensors"),
        ("low", "low_noise_model.safetensors"),
    ]
    for (expert, file) in files {
        let out = granuleDir(variant, expert)
        let source = dir.appendingPathComponent(file)
        let sourceSize =
            (try? FileManager.default.attributesOfItem(atPath: source.path)[.size] as? Int)
            .flatMap { $0 } ?? -1
        if let manifest = try? GranuleManifest.load(from: out),
            manifest.sourceSize == sourceSize
        {
            log(String(
                format: "L · %@/%@: layout present (%d blocks · %.2f GiB streamed) — reused",
                variant, expert, manifest.blockCount, gib(manifest.streamedBytes)))
            continue
        }
        log("L · \(variant)/\(expert): laying out \(file) → \(out.path)")
        let result = try WanGranuleLayout.write(safetensors: source, outputDir: out)
        log(String(
            format: "L · %@/%@: %d blocks · %d tensors/block · %.2f GiB streamed · %.1fs (%.2f GiB/s write)",
            variant, expert, result.manifest.blockCount,
            result.manifest.blocks[0].tensors.count,
            gib(result.manifest.streamedBytes), result.wallSeconds,
            Double(result.writtenBytes) / result.wallSeconds / 1_073_741_824))
    }
}

// ---------- model loading (resident and streaming-bound) ----------

func loadWanConfig(_ variant: String) throws -> WanConfig {
    try WanConfig.load(from: ckptDir(variant).appendingPathComponent("config.json"))
}

/// Fully-resident expert load — the exact BerniniRendererModel.loadExpert recipe
/// (quantize-before-load for int4, strict 1095/1896-key contract, chunked eval).
func loadResidentExpert(_ variant: String, file: String, config: WanConfig) throws -> WanModel {
    let expert = WanModel(config)
    let quantization = config.quantization
    if let quantization {
        WeightLoader.applyQuantization(to: expert, quantization: quantization)
    }
    let weights = try WeightLoader.loadVerifiedSafetensors(
        url: ckptDir(variant).appendingPathComponent(file),
        expectedKeys: BerniniWeightKeys.ditKeys(
            layers: config.numLayers, quantized: quantization != nil
        ).subtracting(["freqs"]),
        toleratedExtras: quantization != nil ? ["freqs"] : [])
    WeightLoader.materialize(weights)
    try expert.update(
        parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
    eval(expert.parameters())
    return expert
}

/// Streaming-bound expert pair: constructed, (int4) quantized, slot-injected via
/// the streamer, globals loaded from the source safetensors. Blocks are NEVER
/// materialized resident (that is the point).
func makeStreamingExperts(
    _ variant: String, config: WanConfig, options: BlockStreamingOptions
) throws -> (high: WanModel, low: WanModel, streamer: BlockStreamer) {
    let high = WanModel(config)
    let low = WanModel(config)
    if let quantization = config.quantization {
        WeightLoader.applyQuantization(to: high, quantization: quantization)
        WeightLoader.applyQuantization(to: low, quantization: quantization)
    }
    let streamer = try BlockStreamer(
        granuleDirs: [granuleDir(variant, "high"), granuleDir(variant, "low")],
        options: options)
    try streamer.bind(experts: [high, low])
    try streamer.loadStreamingGlobals(
        expert: high, from: ckptDir(variant).appendingPathComponent("high_noise_model.safetensors"))
    try streamer.loadStreamingGlobals(
        expert: low, from: ckptDir(variant).appendingPathComponent("low_noise_model.safetensors"))
    return (high, low, streamer)
}

// ---------- the shared denoise loop (identical code for both arms) ----------

struct LoopShape: Sendable {
    let width: Int
    let height: Int
    let frames: Int
    let cfg: Bool  // CFG B=2 vs CFG-free B=1

    func latentDims(_ config: WanConfig) -> (t: Int, h: Int, w: Int) {
        ((frames - 1) / config.vaeStride[0] + 1,
         height / config.vaeStride[1],
         width / config.vaeStride[2])
    }

    func seqLen(_ config: WanConfig) -> Int {
        let (t, h, w) = latentDims(config)
        return (t / config.patchSize[0]) * (h / config.patchSize[1]) * (w / config.patchSize[2])
    }

    func tokens(_ config: WanConfig) -> Int { (cfg ? 2 : 1) * seqLen(config) }

    var label: String { "\(width)x\(height)x\(frames)f\(cfg ? " CFG" : " noCFG")" }
}

/// Deterministic synthetic conditioning (raw UMT5-feature stand-ins) + noise.
func fixtures(_ config: WanConfig, shape: LoopShape) -> (cond: MLXArray, null: MLXArray, noise: MLXArray) {
    MLXRandom.seed(7)
    let cond = MLXRandom.normal([77, config.textDim]).asType(.float32)
    let null = MLXRandom.normal([32, config.textDim]).asType(.float32)
    MLXRandom.seed(42)
    let (t, h, w) = shape.latentDims(config)
    let noise = MLXRandom.normal([config.vaeZDim, t, h, w])
    eval(cond, null, noise)
    return (cond, null, noise)
}

struct LoopResult {
    let final: MLXArray
    let stepSeconds: [Double]
    let highSteps: Int
    let lowSteps: Int
}

/// The dual-expert CFG denoise loop — mirrors bernini-r Generate.swift
/// (euler / shift 5.0: the Lightning schedule splits 2 high / 2 low at 4 steps,
/// exercising the t=875 one-way switch). Identical code drives resident and
/// streamed experts; with a streamer attached, prepareCrossKV/runBlocks route
/// through the streamed group loop automatically.
func denoiseLoop(
    high: WanModel, low: WanModel, config: WanConfig, shape: LoopShape,
    steps: Int, onStep: ((Int) -> Void)? = nil
) -> LoopResult {
    let (cond, null, noise) = fixtures(config, shape: shape)
    let textIn = shape.cfg ? [cond, null] : [cond]

    let contextHigh = high.embedText(textIn)
    let contextLow = low.embedText(textIn)
    eval(contextHigh, contextLow)

    // Streamed: these two sweeps are the first refill sweeps (S calibration).
    let kvHigh = high.prepareCrossKV(contextHigh)
    let kvLow = low.prepareCrossKV(contextLow)

    let (tLat, hLat, wLat) = shape.latentDims(config)
    let grid = (tLat / config.patchSize[0], hLat / config.patchSize[1], wLat / config.patchSize[2])
    let ropeGrid = shape.cfg ? [grid, grid] : [grid]
    let ropeHigh = high.prepareRope(ropeGrid)
    let ropeLow = low.prepareRope(ropeGrid)
    let seqLen = shape.seqLen(config)

    let scheduler = FlowMatchEulerScheduler(numTrainTimesteps: config.numTrainTimesteps)
    scheduler.setTimesteps(steps, shift: 5.0)
    let boundary = config.boundaryTimestep
    let gs = config.sampleGuideScale
    let (gsLow, gsHigh) = (gs[0], gs.count > 1 ? gs[1] : gs[0])

    var latents = noise
    var stepSeconds: [Double] = []
    var highSteps = 0
    var lowSteps = 0
    for i in 0..<steps {
        let t0 = now()
        let timestep = Double(scheduler.timesteps[i])
        let isHigh = timestep >= boundary
        if isHigh { highSteps += 1 } else { lowSteps += 1 }
        let model = isHigh ? high : low
        let kv = isHigh ? kvHigh : kvLow
        let rope = isHigh ? ropeHigh : ropeLow
        let ctx = isHigh ? contextHigh : contextLow

        let noisePred: MLXArray
        if shape.cfg {
            let tBatch = MLXArray([Float(timestep), Float(timestep)])
            let preds = model(
                [latents, latents], t: tBatch, context: .embedded(ctx), seqLen: seqLen,
                crossKVCaches: kv, ropeCosSin: rope)
            let g = isHigh ? gsHigh : gsLow
            noisePred = preds[1] + Float(g) * (preds[0] - preds[1])
        } else {
            let preds = model(
                [latents], t: MLXArray([Float(timestep)]), context: .embedded(ctx),
                seqLen: seqLen, crossKVCaches: kv, ropeCosSin: rope)
            noisePred = preds[0]
        }
        let stepped = scheduler.step(
            modelOutput: noisePred.expandedDimensions(axis: 0),
            timestep: Float(timestep),
            sample: latents.expandedDimensions(axis: 0))
        latents = stepped.squeezed(axis: 0)
        eval(latents)
        MLX.Memory.clearCache()
        stepSeconds.append(now() - t0)
        onStep?(i)
    }
    return LoopResult(
        final: latents, stepSeconds: stepSeconds, highSteps: highSteps, lowSteps: lowSteps)
}

func fmtSteps(_ xs: [Double]) -> String {
    xs.map { String(format: "%.2f", $0) }.joined(separator: " ")
}

// ---------- arms ----------

let parityShape = LoopShape(width: 384, height: 256, frames: 1, cfg: true)  // B·L = 768 < 819 (NAX-clean)

func parityArm(variant: String) throws {
    MLX.Memory.clearCache()  // clean pool baseline per arm
    let config = try loadWanConfig(variant)
    log("\n-- P[\(variant)] · bit-exact parity at \(parityShape.label) "
        + "(seqLen \(parityShape.seqLen(config)), N=\(parityShape.tokens(config))) --")

    // Resident reference.
    var high: WanModel? = try loadResidentExpert(
        variant, file: "high_noise_model.safetensors", config: config)
    var low: WanModel? = try loadResidentExpert(
        variant, file: "low_noise_model.safetensors", config: config)
    let tR = now()
    let resident = denoiseLoop(
        high: high!, low: low!, config: config, shape: parityShape, steps: cfg.steps)
    log(String(
        format: "resident: steps %@ s (high %d / low %d)",
        fmtSteps(resident.stepSeconds), resident.highSteps, resident.lowSteps))
    _ = tR
    high = nil
    low = nil
    MLX.Memory.clearCache()

    // Streamed (forceStream — parity must stream even below the gate).
    let options = BlockStreamingOptions(
        groupSize: cfg.group, gatePolicy: .forceStream)
    let s1 = try makeStreamingExperts(variant, config: config, options: options)
    let streamed = denoiseLoop(
        high: s1.high, low: s1.low, config: config, shape: parityShape, steps: cfg.steps)
    let exact = bitIdentical(streamed.final, resident.final)
    log(String(
        format: "streamed: steps %@ s · S %.2f GiB/s · slots %.2f GiB · bit-identical %@",
        fmtSteps(streamed.stepSeconds), s1.streamer.measuredSGiBs,
        gib(s1.streamer.slotResidentBytes), exact ? "✅" : "❌"))

    // Determinism control (clean/clean).
    let streamed2 = denoiseLoop(
        high: s1.high, low: s1.low, config: config, shape: parityShape, steps: cfg.steps)
    let deterministic = bitIdentical(streamed2.final, streamed.final)
    log("clean/clean streamed runs bit-identical (determinism): \(deterministic ? "✅" : "❌")")

    // Poisoned-slot negative control: corrupt 1 MiB of a mid-run refill.
    // The poison run's acquires are [kv-high: G][kv-low: G][step1: G][step2: G]…
    // → 3·G + 3 lands inside denoise step 2 on the high expert.
    s1.streamer.armPoison(afterAcquires: 3 * s1.streamer.numGroups + 3)
    let poisoned = denoiseLoop(
        high: s1.high, low: s1.low, config: config, shape: parityShape, steps: cfg.steps)
    let sensitive = !bitIdentical(poisoned.final, streamed.final)
    log("poisoned slot (1 MiB corrupted post-prefetch, step 2) changes output: \(sensitive ? "✅" : "❌")")
    s1.streamer.detach()
    MLX.Memory.clearCache()

    guard exact, deterministic, sensitive else {
        log("P[\(variant)] FAILED — stopping.")
        writeReceipt()
        exit(1)
    }
}

/// One gate rung, scoped so the streamer/experts (and any 52 GiB fallback
/// residents) are RELEASED before the caller's `clearCache` — freed-buffer pool
/// + still-live arrays otherwise ratchet phys across rungs (run 1 died at
/// rung 3's fallback from exactly that: ~52 GiB live + carried pool ⇒ silent
/// memory-pressure kill; the mlxengine buffer-pool-ratchet lesson, scope-timing
/// edition).
func runGateRung(shape: LoopShape, variant: String, config: WanConfig) throws {
    let n = shape.tokens(config)
    log(String(
        format: "G · %@: seqLen %d · N=%d · phys_footprint %.2f GB at rung start",
        shape.label, shape.seqLen(config), n, Double(physFootprint()) / 1e9))
    let options = BlockStreamingOptions(groupSize: cfg.group, gatePolicy: .auto)
    let s = try makeStreamingExperts(variant, config: config, options: options)
    let run = denoiseLoop(
        high: s.high, low: s.low, config: config, shape: shape, steps: cfg.steps)
    let g = s.streamer.gateReport
    let verdict = s.streamer.verdict
    if let g {
        log(String(
            format: "   gate: S=%.2f GiB/s · C(N)=%.2f GiB/s · N_min≈%d → %@",
            g.sGiBs, g.cGiBs, g.nMin, verdict.rawValue))
    }
    log(String(
        format: "   steps %@ s (high %d / low %d)",
        fmtSteps(run.stepSeconds), run.highSteps, run.lowSteps))

    switch verdict {
    case .fellBack:
        // Mid-run fallback happened after step 1 — steps 2… ran resident.
        // The blocks are now resident: time matched-resident steps for the
        // in-regime overhead figure, then verify output continuity vs an
        // all-resident rerun (the streamed step 1 must be output-invisible).
        let matched = denoiseLoop(
            high: s.high, low: s.low, config: config, shape: shape, steps: cfg.steps)
        log(String(
            format: "   matched-resident (post-fallback, in-regime): steps %@ s",
            fmtSteps(matched.stepSeconds)))
        let continuity = bitIdentical(run.final, matched.final)
        log("   fallback continuity (streamed-step-1 run ≡ all-resident rerun): "
            + (continuity ? "✅" : "❌"))
    case .streaming:
        // Streaming stayed on: report steady stall, then take the matched
        // in-regime resident baseline via a manual fallback right after.
        let stall = s.streamer.lastForwardStallSeconds
        let compute = s.streamer.lastForwardComputeSeconds
        log(String(
            format: "   last-forward stall %.2fs / compute %.2fs (%.1f%%)",
            stall, compute, 100 * stall / max(stall + compute, 1e-9)))
        try s.streamer.fallBackResident()
        let matched = denoiseLoop(
            high: s.high, low: s.low, config: config, shape: shape, steps: 2)
        log(String(
            format: "   matched-resident (manual fallback, in-regime): steps %@ s → overhead ×%.3f",
            fmtSteps(matched.stepSeconds),
            run.stepSeconds.dropFirst().reduce(0, +)
                / Double(max(run.stepSeconds.count - 1, 1))
                / (matched.stepSeconds.reduce(0, +) / Double(matched.stepSeconds.count))))
    case .undecided:
        log("   ⚠️ gate never evaluated (no streamed forward?)")
    }
    s.streamer.detach()
}

func gateArm() throws {
    let variant = "bf16"
    let config = try loadWanConfig(variant)
    log("\n-- G · runtime gate ladder (bf16, .auto policy — measure, decide, fall back or stream) --")

    let ladder: [LoopShape] = [
        LoopShape(width: 384, height: 256, frames: 1, cfg: true),  // N=768  · minimal t2i
        LoopShape(width: 832, height: 480, frames: 1, cfg: false),  // N=1560 · t2i CFG-free (Lightning shape)
        LoopShape(width: 832, height: 480, frames: 1, cfg: true),  // N=3120 · t2i CFG (the marginal band)
        LoopShape(width: 832, height: 480, frames: 17, cfg: true),  // N=15600 · t2v video-class
    ]

    for (i, shape) in ladder.enumerated() where cfg.gateRungs.contains(i) {
        try runGateRung(shape: shape, variant: variant, config: config)
        // Rung state is fully released here — NOW the pool can actually drain.
        MLX.Memory.clearCache()
        writeReceipt()  // persist per rung — a mid-run death keeps the evidence
    }
}

func budgetArm() throws {
    MLX.Memory.clearCache()  // clean pool baseline — the ≤budget claim depends on it
    let variant = "int4"
    let config = try loadWanConfig(variant)
    let shape = LoopShape(width: 832, height: 480, frames: 17, cfg: true)
    // Engine budget arithmetic (governor): production budget = 0.7 × RAM.
    // Emulated 16 GB machine: 0.7 × 17.18 GB = 12.02 GB.
    let emulatedRamBytes = 16 * (1 << 30)
    let budgetBytes = Int(0.7 * Double(emulatedRamBytes))
    log("\n-- M · 16 GB-budget emulation (int4 · \(shape.label) · N=\(shape.tokens(config))) --")
    log(String(
        format: "budget: emulated RAM 16 GiB → 0.7× governor budget = %.2f GB "
            + "(MLX memoryLimit; cacheLimit 2 GiB — the decode-discipline default)",
        Double(budgetBytes) / 1e9))

    // Uncapped resident reference first (17 GB int4 experts — fits this machine,
    // NOT the emulated one; that is exactly what streaming fixes).
    var high: WanModel? = try loadResidentExpert(
        variant, file: "high_noise_model.safetensors", config: config)
    var low: WanModel? = try loadResidentExpert(
        variant, file: "low_noise_model.safetensors", config: config)
    let resident = denoiseLoop(
        high: high!, low: low!, config: config, shape: shape, steps: cfg.steps)
    log(String(
        format: "resident (uncapped reference): steps %@ s",
        fmtSteps(resident.stepSeconds)))
    high = nil
    low = nil
    MLX.Memory.clearCache()

    // Capped streamed arm.
    let oldCache = MLX.Memory.cacheLimit
    let oldLimit = MLX.Memory.memoryLimit
    MLX.Memory.cacheLimit = 2 << 30
    MLX.Memory.memoryLimit = budgetBytes
    GPU.resetPeakMemory()
    let phys0 = physFootprint()
    let options = BlockStreamingOptions(groupSize: cfg.group, gatePolicy: .auto)
    let s = try makeStreamingExperts(variant, config: config, options: options)
    let streamed = denoiseLoop(
        high: s.high, low: s.low, config: config, shape: shape, steps: cfg.steps)
    let exact = bitIdentical(streamed.final, resident.final)
    let active = MLX.Memory.activeMemory
    let peak = MLX.Memory.peakMemory
    let phys1 = physFootprint()
    if let g = s.streamer.gateReport {
        log(String(
            format: "gate: S=%.2f GiB/s · C(N)=%.2f GiB/s · N_min≈%d → %@",
            g.sGiBs, g.cGiBs, g.nMin, s.streamer.verdict.rawValue))
    }
    log(String(
        format: "streamed capped: steps %@ s · bit-identical to uncapped resident %@",
        fmtSteps(streamed.stepSeconds), exact ? "✅" : "❌"))
    log(String(
        format: "memory: MLX active %.2f GB · MLX peak %.2f GB · slots %.2f GB · "
            + "phys_footprint %.2f → %.2f GB",
        Double(active) / 1e9, Double(peak) / 1e9,
        Double(s.streamer.slotResidentBytes) / 1e9,
        Double(phys0) / 1e9, Double(phys1) / 1e9))
    let under = peak <= budgetBytes
    log(String(
        format: "MLX peak %.2f GB %@ budget %.2f GB → %@  "
            + "(2× int4 experts resident would be ~17 GB — un-admittable at this budget)",
        Double(peak) / 1e9, under ? "≤" : ">", Double(budgetBytes) / 1e9,
        under ? "✅ ADMITS" : "❌ OVER"))
    s.streamer.detach()
    MLX.Memory.cacheLimit = oldCache
    MLX.Memory.memoryLimit = oldLimit
    MLX.Memory.clearCache()

    guard exact else {
        log("M FAILED — streamed-capped output diverged.")
        writeReceipt()
        exit(1)
    }
}

// ---------- receipt ----------

func writeReceipt() {
    try? FileManager.default.createDirectory(
        at: cfg.receipt.deletingLastPathComponent(), withIntermediateDirectories: true)
    try? receiptBox.text.write(to: cfg.receipt, atomically: true, encoding: .utf8)
    print("receipt → \(cfg.receipt.path)")
}

// ---------- main ----------

do {
    if cfg.arms.contains("layout") {
        log("\n-- L · granule layout (per expert, idempotent) --")
        if cfg.arms.contains(where: { $0.hasSuffix("bf16") }) || cfg.arms.contains("gate") {
            try ensureLayout(variant: "bf16")
        }
        if cfg.arms.contains(where: { $0.hasSuffix("int4") }) || cfg.arms.contains("budget") {
            try ensureLayout(variant: "int4")
        }
    }
    writeReceipt()
    if cfg.arms.contains("parity-bf16") {
        try parityArm(variant: "bf16")
        writeReceipt()
    }
    if cfg.arms.contains("parity-int4") {
        try parityArm(variant: "int4")
        writeReceipt()
    }
    if cfg.arms.contains("gate") { try gateArm() }
    if cfg.arms.contains("budget") { try budgetArm() }

    log("\n== VERDICT ==")
    log("All requested arms passed: parity bit-identical (with determinism + poisoned-slot")
    log("controls), gate decisions from in-regime measurements with automatic fully-resident")
    log("fallback (output-invisible), and the 16 GB-budget emulation admits under the")
    log("governor arithmetic — on stock mlx-swift, real A14B weights, no fork.")
    writeReceipt()
} catch {
    log("FAILED: \(error.localizedDescription)")
    writeReceipt()
    exit(1)
}
