// WanConfig — the RESOLVED Wan2.2-T2V-A14B config, decoded from the converted
// checkpoint's config.json (which the Python oracle's conversion wrote fully
// resolved — no parent-class defaults are missing; see mlx-porting pitfall #10).
// Field set mirrors mlx_video/models/wan_2/config.py `WanModelConfig`.

import Foundation

/// Decodes a JSON field that is EITHER a scalar number OR an array of numbers into
/// `[Double]`, transparently — so consumers still see `[Double]`. Wan2.2 ships both
/// shapes for `sample_guide_scale`: A14B is `[high, low]` (dual expert), TI2V-5B is a
/// bare `5.0` (single expert). Encodes back as a scalar when there's exactly one value.
@propertyWrapper
public struct ScalarOrArrayDouble: Codable, Sendable, Equatable {
    public var wrappedValue: [Double]
    public init(wrappedValue: [Double]) { self.wrappedValue = wrappedValue }

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let scalar = try? c.decode(Double.self) {
            wrappedValue = [scalar]
        } else {
            wrappedValue = try c.decode([Double].self)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        if wrappedValue.count == 1 {
            try c.encode(wrappedValue[0])
        } else {
            try c.encode(wrappedValue)
        }
    }
}

public struct WanQuantization: Codable, Sendable, Equatable {
    public var groupSize: Int
    public var bits: Int

    public init(groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
    }
}

public struct WanConfig: Codable, Sendable, Equatable {
    public var modelType: String
    public var modelVersion: String
    public var patchSize: [Int]
    public var textLen: Int
    public var inDim: Int
    public var dim: Int
    public var ffnDim: Int
    public var freqDim: Int
    public var textDim: Int
    public var outDim: Int
    public var numHeads: Int
    public var numLayers: Int
    public var windowSize: [Int]
    public var qkNorm: Bool
    public var crossAttnNorm: Bool
    public var eps: Double
    public var vaeStride: [Int]
    public var vaeZDim: Int
    public var dualModel: Bool
    public var boundary: Double
    public var sampleShift: Double
    public var sampleSteps: Int
    @ScalarOrArrayDouble public var sampleGuideScale: [Double]
    public var numTrainTimesteps: Int
    public var sampleFps: Int
    public var frameNum: Int
    public var sampleNegPrompt: String
    public var maxArea: Int
    public var t5VocabSize: Int
    public var t5Dim: Int
    public var t5DimAttn: Int
    public var t5DimFfn: Int
    public var t5NumHeads: Int
    public var t5NumLayers: Int
    public var t5NumBuckets: Int
    /// Present only in quantized checkpoints (int4: group_size 64, bits 4).
    public var quantization: WanQuantization?

    public var headDim: Int { dim / numHeads }

    /// Absolute timestep of the high/low expert boundary (875.0 at A14B defaults).
    public var boundaryTimestep: Double { boundary * Double(numTrainTimesteps) }

    /// Explicit public memberwise init (a public struct's implicit memberwise init is
    /// `internal`, so cross-module consumers — tests, Helios, Phantom — need this).
    public init(
        modelType: String, modelVersion: String, patchSize: [Int], textLen: Int,
        inDim: Int, dim: Int, ffnDim: Int, freqDim: Int, textDim: Int, outDim: Int,
        numHeads: Int, numLayers: Int, windowSize: [Int], qkNorm: Bool,
        crossAttnNorm: Bool, eps: Double, vaeStride: [Int], vaeZDim: Int,
        dualModel: Bool, boundary: Double, sampleShift: Double, sampleSteps: Int,
        sampleGuideScale: [Double], numTrainTimesteps: Int, sampleFps: Int,
        frameNum: Int, sampleNegPrompt: String, maxArea: Int, t5VocabSize: Int,
        t5Dim: Int, t5DimAttn: Int, t5DimFfn: Int, t5NumHeads: Int,
        t5NumLayers: Int, t5NumBuckets: Int, quantization: WanQuantization?
    ) {
        self.modelType = modelType; self.modelVersion = modelVersion
        self.patchSize = patchSize; self.textLen = textLen
        self.inDim = inDim; self.dim = dim; self.ffnDim = ffnDim; self.freqDim = freqDim
        self.textDim = textDim; self.outDim = outDim
        self.numHeads = numHeads; self.numLayers = numLayers
        self.windowSize = windowSize; self.qkNorm = qkNorm
        self.crossAttnNorm = crossAttnNorm; self.eps = eps
        self.vaeStride = vaeStride; self.vaeZDim = vaeZDim
        self.dualModel = dualModel; self.boundary = boundary
        self.sampleShift = sampleShift; self.sampleSteps = sampleSteps
        self.sampleGuideScale = sampleGuideScale; self.numTrainTimesteps = numTrainTimesteps
        self.sampleFps = sampleFps; self.frameNum = frameNum
        self.sampleNegPrompt = sampleNegPrompt; self.maxArea = maxArea
        self.t5VocabSize = t5VocabSize; self.t5Dim = t5Dim
        self.t5DimAttn = t5DimAttn; self.t5DimFfn = t5DimFfn
        self.t5NumHeads = t5NumHeads; self.t5NumLayers = t5NumLayers
        self.t5NumBuckets = t5NumBuckets; self.quantization = quantization
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case modelVersion = "model_version"
        case patchSize = "patch_size"
        case textLen = "text_len"
        case inDim = "in_dim"
        case dim
        case ffnDim = "ffn_dim"
        case freqDim = "freq_dim"
        case textDim = "text_dim"
        case outDim = "out_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case windowSize = "window_size"
        case qkNorm = "qk_norm"
        case crossAttnNorm = "cross_attn_norm"
        case eps
        case vaeStride = "vae_stride"
        case vaeZDim = "vae_z_dim"
        case dualModel = "dual_model"
        case boundary
        case sampleShift = "sample_shift"
        case sampleSteps = "sample_steps"
        case sampleGuideScale = "sample_guide_scale"
        case numTrainTimesteps = "num_train_timesteps"
        case sampleFps = "sample_fps"
        case frameNum = "frame_num"
        case sampleNegPrompt = "sample_neg_prompt"
        case maxArea = "max_area"
        case t5VocabSize = "t5_vocab_size"
        case t5Dim = "t5_dim"
        case t5DimAttn = "t5_dim_attn"
        case t5DimFfn = "t5_dim_ffn"
        case t5NumHeads = "t5_num_heads"
        case t5NumLayers = "t5_num_layers"
        case t5NumBuckets = "t5_num_buckets"
        case quantization
    }

    public static func load(from url: URL) throws -> WanConfig {
        try JSONDecoder().decode(WanConfig.self, from: Data(contentsOf: url))
    }
}

extension WanQuantization {
    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}
