// WanStreaming.swift — the Wan adapter over BlockStreamKit
// (mlx-block-stream-swift; NEUROSTREAM-ACTIONS HV2, extraction phase).
//
// The in-tree GranuleLayout/BlockStreamer implementation (b45b879, receipts
// probes/hv2_wan_blockstreamer.out) moved into the shared substrate once the
// second consumer (ltx-2-mlx-swift bf0a2ba) pinned the seam. What stays HERE
// is the irreducibly Wan-specific ~20%:
//   - the checkpoint dialect (`blocks.` prefix, WanAttentionBlock forward-order
//     ranking for granule locality),
//   - the MODULE-shaped bind: `update(parameters:)` injection into every
//     expert's blocks (streamed slot arrays are SHARED across experts; the kit
//     v0.2.0 multi-set support was built for exactly this),
//   - `loadStreamingGlobals` (Module update + CPU-stream materialize),
//   - expert activation (`ensureActive(for:)` — the A14B t=875 switch),
//   - fallback/detach wiring through `WanModel.blockStreamer`.
//
// Load semantics: `castPolicy: .none` — Wan loads raw checkpoint dtypes (no
// resident-init cast exists to mirror), so streamed and bind-resident tensors
// alike carry granule bytes verbatim, exactly as the pre-extraction
// implementation did. Public API is 1:1 with b45b879 so WanModel's routed
// loops, RunWanStream, the unit gates, and bernini-r's consumer wiring
// (d02cfa1) compile unchanged.

@_exported import BlockStreamKit
import Foundation
import MLX
import MLXNN

/// Re-exports so consumers keep one import.
public typealias BlockStreamingOptions = BlockStreamKit.BlockStreamingOptions
public typealias BlockStreamerGateReport = BlockStreamKit.StreamGateReport
public typealias BlockStreamerError = BlockStreamKit.BlockStreamError
public typealias GranuleManifest = BlockStreamKit.GranuleManifest
public typealias GranuleTensor = BlockStreamKit.GranuleTensor
public typealias SafetensorsHeader = BlockStreamKit.SafetensorsHeader

public enum WanGranuleLayout {
    /// Canonical within-block tensor order: WanAttentionBlock forward order.
    /// Locality only — correctness never depends on it.
    static let forwardOrderPrefixes: [String] = [
        "modulation",
        "norm1.",
        "self_attn.norm_q", "self_attn.norm_k",
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "norm3",
        "cross_attn.norm_q", "cross_attn.norm_k",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        "norm2.",
        "ffn.fc1", "ffn.fc2",
    ]

    public static func forwardRank(_ key: String) -> Int {
        for (i, p) in forwardOrderPrefixes.enumerated() where key.hasPrefix(p) { return i }
        return forwardOrderPrefixes.count
    }

    @discardableResult
    public static func write(
        safetensors: URL, outputDir: URL,
        blockPrefix: String = "blocks.",
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> BlockStreamKit.GranuleLayout.Result {
        try BlockStreamKit.GranuleLayout.write(
            safetensors: safetensors, outputDir: outputDir, blockPrefix: blockPrefix,
            orderRank: forwardRank, progress: progress)
    }
}

/// The Wan streaming plane: a thin adapter that owns the kit core plus the
/// Module-shaped bind. API-compatible with the pre-extraction implementation.
public final class BlockStreamer: @unchecked Sendable {

    public typealias Verdict = BlockStreamKit.BlockStreamer.Verdict

    public let core: BlockStreamKit.BlockStreamer
    private var experts: [WanModel] = []
    private var expertIndex: [ObjectIdentifier: Int] = [:]

    /// One granule directory per expert (dense models pass one; A14B passes
    /// [high, low]). All manifests must describe identical block layouts.
    public init(granuleDirs: [URL], options: BlockStreamingOptions = .init()) throws {
        self.core = try BlockStreamKit.BlockStreamer(granuleDirs: granuleDirs, options: options)
    }

    // MARK: forwarded surface (1:1 with b45b879)

    public var blockCount: Int { core.blockCount }
    public var groupSize: Int { core.groupSize }
    public var numGroups: Int { core.numGroups }
    public var sweepBytes: Int { core.sweepBytes }
    public var slotResidentBytes: Int { core.slotResidentBytes }
    public var verdict: Verdict { core.verdict }
    public var gateReport: BlockStreamerGateReport? { core.gateReport }
    public var measuredSGiBs: Double { core.measuredSGiBs }
    public var lastForwardComputeSeconds: Double { core.lastForwardComputeSeconds }
    public var lastForwardStallSeconds: Double { core.lastForwardStallSeconds }
    var manifests: [GranuleManifest] { core.manifests }

    public func blockRange(_ group: Int) -> Range<Int> { core.blockRange(group) }
    @discardableResult func acquireGroup() -> Int { core.acquireGroup() }
    func releaseGroup() { core.releaseGroup() }
    func beginForward(tokens: Int) { core.beginForward(tokens: tokens) }
    func endForward() { core.endForward() }
    public func finish() { core.finish() }
    public func armPoison(
        afterAcquires: Int, localBlock: Int = 0, tensor: Int = 0, bytes: Int = 1 << 20
    ) {
        core.armPoison(
            afterAcquires: afterAcquires, localBlock: localBlock, tensor: tensor, bytes: bytes)
    }
    public func fallBackResident() throws { try core.fallBackResident() }

    /// Stop IO and detach from every bound expert WITHOUT loading anything
    /// resident — the receipts' between-arms reset. Block parameters keep
    /// aliasing this streamer's slots until another bind replaces them, so no
    /// forward may run between `detach()` and the next bind.
    public func detach() {
        core.detach()
    }

    // MARK: - The Wan bind (Module injection across experts)

    /// Inject slot-backed parameter arrays into every bound expert's blocks
    /// (once — the P-C seam). Experts must be freshly constructed (and, for
    /// quantized checkpoints, already passed through
    /// `WeightLoader.applyQuantization`) with their block parameters UNLOADED;
    /// global tensors load separately via `loadStreamingGlobals`.
    public func bind(experts boundExperts: [WanModel]) throws {
        guard boundExperts.count == core.manifests.count else {
            throw BlockStreamerError.contract(
                "\(boundExperts.count) experts vs \(core.manifests.count) granule sets")
        }
        guard experts.isEmpty else {
            throw BlockStreamerError.state("bind() called twice")
        }

        // Contract check: the model's per-block parameter set must equal the
        // manifest's tensor table — 0 missing / 0 unused, shapes equal (the
        // refuse-partial-loads doctrine, streamed edition). Dtypes come from
        // the manifest (construction dtype differs pre-load, exactly as with
        // `update(parameters:)` on a resident load).
        let template = core.manifests[0].blocks[0].tensors
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

        // Kit bind: raw-load semantics (.none — Wan never casts at load;
        // computeDtype is inert under that policy). The fallback closure swaps
        // resident arrays into the right expert's block; detach unhooks every
        // expert's routing.
        let perSet = try core.bind(
            computeDtype: .bfloat16,
            castPolicy: .none,
            installResident: { [weak self] set, block, pairs in
                guard let self, set < self.experts.count else { return }
                try self.experts[set].blocks[block].update(
                    parameters: ModuleParameters.unflattened(pairs),
                    verify: [.noUnusedKeys])
            },
            onDetach: { [weak self] in
                guard let self else { return }
                for expert in self.experts {
                    expert.blockStreamer = nil
                }
            })

        // Inject each set's arrays into its expert's blocks, then re-verify
        // pointer stability through the Modules' own storage.
        for (set, expert) in boundExperts.enumerated() {
            for bound in perSet[set] {
                try expert.blocks[bound.block].update(
                    parameters: ModuleParameters.unflattened(bound.arrays),
                    verify: [.noUnusedKeys])
            }
        }
        experts = boundExperts
        expertIndex = Dictionary(
            uniqueKeysWithValues: boundExperts.enumerated().map {
                (ObjectIdentifier($1), $0)
            })
        try core.verifyInstalledSets { [weak self] set, block, key in
            guard let self else { return nil }
            let params = Dictionary(
                uniqueKeysWithValues: self.experts[set].blocks[block].parameters().flattened())
            return params[key]
        }
        for expert in boundExperts {
            expert.blockStreamer = self
        }
    }

    /// Load the NON-block ("global") tensors — embeddings, head, … — from the
    /// source safetensors into a bound expert. Loads exactly the model's
    /// non-block parameter set; manifest-listed extras the model never loads
    /// (the int4 checkpoints' serialized `freqs`) are tolerated and skipped.
    public func loadStreamingGlobals(expert: WanModel, from safetensors: URL) throws {
        guard let index = expertIndex[ObjectIdentifier(expert)] else {
            throw BlockStreamerError.state("expert not bound")
        }
        let blockPrefix = core.manifests[index].blockPrefix
        let wanted = Set(
            expert.parameters().flattened().map(\.0).filter { !$0.hasPrefix(blockPrefix) })
        let available = Set(core.manifests[index].globalKeys)
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
        eval(expert.parameters())  // globals only touch resident arrays; slots hold zeros
    }

    /// Make `model`'s granule set the streamed one, (re)starting the prefetch
    /// thread at group 0. Called automatically by the streamed forward/KV
    /// paths; switching experts (the A14B t=875 boundary) is just a different
    /// `model` arriving here.
    func ensureActive(for model: WanModel) {
        guard let index = expertIndex[ObjectIdentifier(model)] else {
            fatalError("BlockStreamer: forward on an unbound expert")
        }
        core.ensureActive(set: index)
    }
}
