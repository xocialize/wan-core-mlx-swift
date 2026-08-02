import Foundation
import MLX
import MLXNN
import XCTest

@testable import WanCore

/// BlockStreamer machinery gates on a TINY synthetic WanModel — no checkpoints.
/// A random-init model's evaluated parameters ARE a valid checkpoint: save them
/// as safetensors, granule-lay them out, and a streaming-bound second instance
/// must produce BIT-IDENTICAL forwards (streaming changes WHERE weights live,
/// never WHAT they are). Controls per the spikes doctrine: clean/clean
/// determinism, a poisoned refill that MUST diverge, and the .auto gate's
/// fully-resident fallback staying byte-exact. The real-weights receipts
/// (RunWanStream → probes/hv2_wan_blockstreamer.out) cover A14B at scale.
final class BlockStreamerTests: XCTestCase {

    static let tinyConfig = WanConfig(
        modelType: "t2v", modelVersion: "test", patchSize: [1, 2, 2], textLen: 16,
        inDim: 16, dim: 128, ffnDim: 256, freqDim: 64, textDim: 64, outDim: 16,
        numHeads: 2, numLayers: 4, windowSize: [-1, -1], qkNorm: true,
        crossAttnNorm: true, eps: 1e-6, vaeStride: [4, 8, 8], vaeZDim: 16,
        dualModel: false, boundary: 0.875, sampleShift: 5.0, sampleSteps: 4,
        sampleGuideScale: [3.0, 4.0], numTrainTimesteps: 1000, sampleFps: 16,
        frameNum: 1, sampleNegPrompt: "", maxArea: 0, t5VocabSize: 0, t5Dim: 0,
        t5DimAttn: 0, t5DimFfn: 0, t5NumHeads: 0, t5NumLayers: 0, t5NumBuckets: 0,
        quantization: nil)

    /// Deterministic small input: latent [16, 1, 8, 8] → seqLen 16, plus a raw
    /// text-feature fixture.
    static func makeInputs() -> (x: MLXArray, ctx: MLXArray) {
        let x = MLXArray(
            (0..<(16 * 1 * 8 * 8)).map { Float(sin(Double($0) * 0.017)) }, [16, 1, 8, 8])
        let ctx = MLXArray(
            (0..<(5 * 64)).map { Float(cos(Double($0) * 0.031)) }, [5, 64])
        eval(x, ctx)
        return (x, ctx)
    }

    static func forward(_ model: WanModel, _ x: MLXArray, _ ctx: MLXArray) -> MLXArray {
        let embedded = model.embedText([ctx])
        let kv = model.prepareCrossKV(embedded)
        let out = model(
            [x], t: MLXArray([Float(999)]), context: .embedded(embedded), seqLen: 16,
            crossKVCaches: kv)[0]
        eval(out)
        return out
    }

    static func bitIdentical(_ a: MLXArray, _ b: MLXArray) -> Bool {
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

    /// Save a random-init reference model as a safetensors checkpoint, lay out
    /// granules, and return (referenceModel, checkpointURL, granuleDir).
    static func makeFixture() throws -> (WanModel, URL, URL) {
        let reference = WanModel(tinyConfig)
        eval(reference.parameters())

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("wan-blockstreamer-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let ckpt = dir.appendingPathComponent("model.safetensors")
        let flat = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        try MLX.save(arrays: flat, url: ckpt)

        let granules = dir.appendingPathComponent("granules")
        let result = try GranuleLayout.write(safetensors: ckpt, outputDir: granules)
        XCTAssertEqual(result.manifest.blockCount, tinyConfig.numLayers)
        return (reference, ckpt, granules)
    }

    func testStreamedForwardBitIdenticalPlusControls() throws {
        let (reference, ckpt, granules) = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: granules.deletingLastPathComponent()) }
        let (x, ctx) = Self.makeInputs()
        let residentOut = Self.forward(reference, x, ctx)

        let streamed = WanModel(Self.tinyConfig)
        let streamer = try BlockStreamer(
            granuleDirs: [granules],
            options: BlockStreamingOptions(
                groupSize: 2, gatePolicy: .forceStream, quiet: true))
        try streamer.bind(experts: [streamed])
        try streamer.loadStreamingGlobals(expert: streamed, from: ckpt)

        // Parity: streamed ≡ resident, bit-for-bit.
        let streamedOut = Self.forward(streamed, x, ctx)
        XCTAssertTrue(
            Self.bitIdentical(streamedOut, residentOut),
            "streamed forward must be bit-identical to resident")

        // Determinism control: clean/clean streamed runs identical.
        let streamedOut2 = Self.forward(streamed, x, ctx)
        XCTAssertTrue(
            Self.bitIdentical(streamedOut2, streamedOut),
            "clean/clean streamed runs must be bit-identical")

        // Poisoned-slot negative control: the compare must have teeth. The
        // poison-run's acquires are [kv sweep: numGroups][runBlocks: numGroups];
        // target the FIRST runBlocks group so a forward actually reads the
        // corrupted tensor (tensor 0 = modulation in granule forward order).
        streamer.armPoison(
            afterAcquires: streamer.numGroups, localBlock: 0, tensor: 0, bytes: 256)
        let poisonedOut = Self.forward(streamed, x, ctx)
        XCTAssertFalse(
            Self.bitIdentical(poisonedOut, streamedOut),
            "a poisoned refill MUST change the output")

        // Clean again after the one-shot poison (slots refill every sweep).
        let streamedOut3 = Self.forward(streamed, x, ctx)
        XCTAssertTrue(
            Self.bitIdentical(streamedOut3, streamedOut),
            "output must recover once the poisoned slot is refilled")

        streamer.finish()
    }

    func testAutoGateFallsBackResidentAndStaysExact() throws {
        let (reference, ckpt, granules) = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: granules.deletingLastPathComponent()) }
        let (x, ctx) = Self.makeInputs()
        let residentOut = Self.forward(reference, x, ctx)

        let streamed = WanModel(Self.tinyConfig)
        // At toy scale IO trivially keeps up with dispatch-bound compute, so
        // the measured gate can legitimately clear — force the fallback branch
        // deterministically with an unreachable margin (the measurement path
        // still runs; only the threshold is rigged).
        let streamer = try BlockStreamer(
            granuleDirs: [granules],
            options: BlockStreamingOptions(
                groupSize: 2, gatePolicy: .auto, gateMargin: 1e12, quiet: true))
        try streamer.bind(experts: [streamed])
        try streamer.loadStreamingGlobals(expert: streamed, from: ckpt)

        // The gating forward itself must stream bit-exactly, then fall back.
        let firstOut = Self.forward(streamed, x, ctx)
        XCTAssertTrue(
            Self.bitIdentical(firstOut, residentOut),
            "the gating forward itself must already be bit-identical")
        XCTAssertEqual(
            streamer.verdict, .fellBack,
            "the rigged gate must fail and fall back resident")
        XCTAssertNil(
            streamed.blockStreamer,
            "fallback must detach the streamer from the model")

        // Post-fallback (resident) forwards remain bit-identical.
        let residentAgain = Self.forward(streamed, x, ctx)
        XCTAssertTrue(
            Self.bitIdentical(residentAgain, residentOut),
            "post-fallback resident forward must be bit-identical")
    }
}
