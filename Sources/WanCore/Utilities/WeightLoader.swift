//
//  WeightLoader.swift
//
//  LIFTED from the Swift component donor:
//    /Volumes/DEV_ARCHIVE/longcat-avatar-mlx-swift
//      Sources/LongCatVideoAvatar/Utilities/WeightLoader.swift
//  Adapted for Bernini-R's FLAT checkpoint layout (no component subdirs):
//  `{high_noise_model,low_noise_model,vae,t5_encoder}.safetensors` +
//  `config.json` in one directory.
//
//  Runtime HF download + cache helper. No Python equivalent — this is the
//  Swift port's substitute for Python's `huggingface_hub.snapshot_download`
//  and `mx.load`. We ship our own minimal HF Hub client (URLSession +
//  the public /api/models/ REST endpoint) instead of depending on
//  swift-transformers' internal `Hub` target, which isn't exposed as a
//  library product.
//
//  Standard repo IDs (point at the same artifacts as the Python oracle):
//    - mlx-community/Bernini-R-bf16   (~64 GB; both experts resident)
//    - mlx-community/Bernini-R-int4   (~27 GB; the consumer config)
//
//  Env override:
//    BERNINI_R_WEIGHTS_DIR=/path/to/local/ckpt-dir/   (e.g. the DEV_ARCHIVE
//    mirror's ckpt-bf16/ or ckpt-int4/; bypasses download entirely)
//

import Foundation
import MLX
import MLXNN

public enum WeightLoaderError: LocalizedError {
    case invalidRepoID(String)
    case httpError(URL, status: Int)
    case decodingError(URL, underlying: Error)
    case missingComponent(component: String, in: URL)
    case keyContractViolation(URL, missing: [String], unused: [String])

    public var errorDescription: String? {
        switch self {
        case .invalidRepoID(let r):
            return "Invalid HF repo id: \(r). Expected <org>/<name>."
        case .httpError(let url, let s):
            return "HTTP \(s) for \(url.absoluteString)"
        case .decodingError(let url, let err):
            return "Failed to decode JSON at \(url.lastPathComponent): \(err.localizedDescription)"
        case .missingComponent(let c, let dir):
            return "Component '\(c).safetensors' not found under \(dir.path) "
                + "(expected flat layout: high_noise_model/low_noise_model/vae/t5_encoder "
                + ".safetensors + config.json)"
        case .keyContractViolation(let url, let missing, let unused):
            return "Key contract violation for \(url.lastPathComponent): "
                + "\(missing.count) missing (e.g. \(missing.sorted().prefix(5))), "
                + "\(unused.count) unused (e.g. \(unused.sorted().prefix(5)))"
        }
    }
}

/// Index file describing a sharded safetensors checkpoint (HuggingFace
/// convention). The published Bernini checkpoints are single-file per
/// component, but the loader keeps shard support for forward compatibility.
public struct SafetensorsIndex: Codable, Sendable {
    public struct Metadata: Codable, Sendable {
        public let totalSize: Int?
        enum CodingKeys: String, CodingKey { case totalSize = "total_size" }
    }
    public let metadata: Metadata?
    public let weightMap: [String: String]   // tensor name → shard filename

    enum CodingKeys: String, CodingKey {
        case metadata
        case weightMap = "weight_map"
    }
}

public enum WeightLoader {

    // MARK: - Cache location

    /// Root cache directory: `<Caches>/bernini-r-mlx-swift/<org>/<repo>/`.
    /// Overridable via `BERNINI_R_WEIGHTS_DIR` env var (points directly
    /// at an unpacked checkpoint dir; bypasses download entirely).
    public static func cacheDirectory(for repoID: String) throws -> URL {
        if let override = ProcessInfo.processInfo.environment["BERNINI_R_WEIGHTS_DIR"],
           !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        let parts = repoID.split(separator: "/")
        guard parts.count == 2 else { throw WeightLoaderError.invalidRepoID(repoID) }
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return caches
            .appendingPathComponent("bernini-r-mlx-swift", isDirectory: true)
            .appendingPathComponent(String(parts[0]), isDirectory: true)
            .appendingPathComponent(String(parts[1]), isDirectory: true)
    }

    // MARK: - HF Hub download

    /// Snapshot-download a complete HF repo into the local cache. Resumable
    /// per-file: already-present files with matching size are skipped. No
    /// xet / git-lfs negotiation — just direct downloads via the public
    /// `resolve/main/<path>` endpoint, which is what `huggingface_hub`
    /// falls back to anyway when xet isn't available.
    @discardableResult
    public static func snapshotDownload(
        repoID: String,
        progress: (@Sendable (_ file: String, _ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> URL {
        // If env override is set, assume the dir already exists with weights
        // and skip network entirely.
        if ProcessInfo.processInfo.environment["BERNINI_R_WEIGHTS_DIR"] != nil {
            return try cacheDirectory(for: repoID)
        }

        let cacheDir = try cacheDirectory(for: repoID)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let files = try await listRepoFiles(repoID: repoID)
        var doneCount = 0
        for entry in files {
            let dest = cacheDir.appendingPathComponent(entry.path)
            if let existing = try? FileManager.default.attributesOfItem(atPath: dest.path),
               let size = existing[.size] as? Int,
               let expected = entry.size, size == expected {
                doneCount += 1
                progress?(entry.path, doneCount, files.count)
                continue
            }
            try FileManager.default.createDirectory(
                at: dest.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try await downloadFile(repoID: repoID, path: entry.path, to: dest)
            doneCount += 1
            progress?(entry.path, doneCount, files.count)
        }
        return cacheDir
    }

    // MARK: - Safetensors loading

    /// Load every shard listed in a safetensors index into a single
    /// `[String: MLXArray]` dict. Loads each shard once. (Unused by the
    /// published single-file-per-component Bernini checkpoints; kept from
    /// the donor for forward compatibility.)
    public static func loadShardedSafetensors(indexURL: URL) throws -> [String: MLXArray] {
        let data = try Data(contentsOf: indexURL)
        let index: SafetensorsIndex
        do {
            index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        } catch {
            throw WeightLoaderError.decodingError(indexURL, underlying: error)
        }

        let baseDir = indexURL.deletingLastPathComponent()
        var grouped: [String: [String]] = [:]
        for (tensor, shard) in index.weightMap {
            grouped[shard, default: []].append(tensor)
        }

        var combined: [String: MLXArray] = [:]
        combined.reserveCapacity(index.weightMap.count)
        for (shard, tensors) in grouped {
            let shardURL = baseDir.appendingPathComponent(shard)
            let shardDict = try MLX.loadArrays(url: shardURL)
            for tensor in tensors {
                guard let arr = shardDict[tensor] else { continue }
                combined[tensor] = arr
            }
        }
        return combined
    }

    /// Convenience: load a single safetensors file.
    public static func loadSafetensors(url: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url)
    }

    /// Materialize lazy loaded tensors in bounded chunks. Evaluating a whole
    /// 28.6 GB expert in one `eval` keeps a single Metal command buffer alive
    /// past the ~10 s GPU timeout (kIOGPUCommandBufferCallbackErrorTimeout,
    /// observed on the first GPU smoke 2026-06-12) — chunking bounds each
    /// buffer. No-op for already-materialized arrays.
    public static func materialize(
        _ weights: [String: MLXArray], chunk: Int = 64
    ) {
        let values = Array(weights.values)
        var i = 0
        while i < values.count {
            let j = min(i + chunk, values.count)
            eval(Array(values[i..<j]))
            i = j
        }
    }

    /// Load a flat single-file safetensors checkpoint and enforce the pinned
    /// key contract: after dropping `toleratedExtras` (conversion-time
    /// artifacts the model never loads — for the int4 experts that is exactly
    /// the serialized `freqs` rope table), the on-disk key set must equal
    /// `expectedKeys` — 0 missing / 0 unused, or the load throws. This is the
    /// refuse-partial-loads gate; drift here is how silent zero-weights ship.
    public static func loadVerifiedSafetensors(
        url: URL,
        expectedKeys: Set<String>,
        toleratedExtras: Set<String> = []
    ) throws -> [String: MLXArray] {
        // Load AND materialize on the CPU stream: lazy Load ops bind to the
        // stream current at creation, and on the GPU stream a multi-GB read
        // from disk keeps one Metal command buffer alive past the ~10 s GPU
        // timeout (kIOGPUCommandBufferCallbackErrorTimeout — first GPU smoke,
        // 2026-06-12; chunked GPU-side eval was NOT sufficient on the archive
        // disk). CPU-stream arrays live in unified memory; GPU kernels
        // consume them directly.
        var weights = try Device.withDefaultDevice(.cpu) {
            let loaded = try MLX.loadArrays(url: url)
            materialize(loaded)
            return loaded
        }
        for extra in toleratedExtras {
            weights.removeValue(forKey: extra)
        }
        let present = Set(weights.keys)
        let missing = expectedKeys.subtracting(present)
        let unused = present.subtracting(expectedKeys)
        guard missing.isEmpty, unused.isEmpty else {
            throw WeightLoaderError.keyContractViolation(
                url, missing: Array(missing), unused: Array(unused))
        }
        return weights
    }

    // MARK: - Config + quantization detection

    /// Read a JSON config file as a Codable type.
    public static func loadConfig<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        let data = try Data(contentsOf: url)
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw WeightLoaderError.decodingError(url, underlying: error)
        }
    }

    /// Returns the quantization config if the published variant carries one
    /// (int4: `{"group_size": 64, "bits": 4}` in the checkpoint root's
    /// `config.json`), or `nil` for the bf16 variant. Thin wrapper over the
    /// `WanConfig.quantization` field for callers that haven't decoded the
    /// full config yet.
    public static func detectQuantization(configURL: URL) throws -> WanQuantization? {
        try WanConfig.load(from: configURL).quantization
    }

    // MARK: - Expert quantize-on-load

    /// Apply `MLXNN.quantize` to a freshly-constructed DiT expert *before*
    /// loading the bit-packed tensors. Required so that `QuantizedLinear`
    /// modules are installed in the right places before
    /// `weight`/`scales`/`biases` land via `update(parameters:)`.
    ///
    /// Mirrors the Python oracle's quantize-before-load and its
    /// `scripts/quantize.py` scope: int4 g64, **Linear-only**, and only the
    /// per-block attention q/k/v/o + ffn.fc1/fc2 projections — embeddings,
    /// norms, modulation, and the head stay bf16. `includePaths` defaults to
    /// exactly that scope (`BerniniWeightKeys.blockLinearPaths`, matched as
    /// path suffixes like `self_attn.q`); `skipPatterns` keeps the donor's
    /// substring-exclusion support for variants that ship one.
    public static func applyQuantization(
        to model: Module,
        quantization: WanQuantization,
        preMaterialize: Bool = true,
        includePaths: [String]? = nil,
        skipPatterns: [String] = []
    ) {
        let include = includePaths ?? BerniniWeightKeys.blockLinearPaths

        // Zero-fill the model's parameters BEFORE quantize. The quantize pass
        // only needs the right SHAPES (every produced value is replaced by the
        // loaded bit-packed tensors); leaving the random inits in place means
        // quantize chains over ~57 GB of lazy fp32 normals — materializing or
        // evaluating them under an already-loaded sibling expert drives the
        // machine into memory pressure and a Metal watchdog kill
        // (kIOGPUCommandBufferCallbackErrorTimeout, S6 2026-06-12, twice).
        // Zeros are constant-fill: near-free to create and evaluate.
        if preMaterialize {
            let flat = model.parameters().flattened()
            let zeroed = Dictionary(
                uniqueKeysWithValues: flat.map {
                    ($0.0, MLXArray.zeros($0.1.shape, dtype: $0.1.dtype))
                })
            try? model.update(parameters: ModuleParameters.unflattened(zeroed))
            // Do NOT eval the zeros: constant-fill graphs have no slow
            // dependencies (no fence stall), and every value is replaced by
            // the loaded tensors before anything evaluates them. Eagerly
            // materializing them costs ~57 GB fp32 per expert (observed as a
            // 161 GB swap-storm peak on the first int4 smoke).
        }

        MLXNN.quantize(
            model: model,
            groupSize: quantization.groupSize,
            bits: quantization.bits,
            filter: { (path: String, module: Module) -> Bool in
                // Only quantize Linear — everything else stays full precision.
                guard module is MLXNN.Linear else { return false }
                // Only the per-block linears the conversion recipe quantized.
                guard include.contains(where: { path.hasSuffix($0) }) else { return false }
                // Skip any path matching a configured skip pattern.
                for pat in skipPatterns where path.contains(pat) {
                    return false
                }
                return true
            }
        )
    }

    // MARK: - Component path helpers

    /// Resolve `<weightsRoot>/<component>.safetensors` and verify it exists.
    /// Components match the published flat checkpoint layout:
    ///   high_noise_model / low_noise_model / vae / t5_encoder
    /// (plus `config.json` alongside).
    public static func componentFile(_ component: String, under root: URL) throws -> URL {
        let url = root.appendingPathComponent("\(component).safetensors", isDirectory: false)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WeightLoaderError.missingComponent(component: component, in: root)
        }
        return url
    }
}

// MARK: - Minimal HF Hub REST client (file-private)

/// One file in the HF repo tree.
private struct HFTreeEntry: Sendable {
    let path: String
    let size: Int?
}

extension WeightLoader {
    /// `GET https://huggingface.co/api/models/{repo}/tree/main?recursive=true`
    /// returns a JSON array of file entries with `path`, `size`, `type`.
    /// We filter to leaf files (type == "file") and return paths + sizes.
    fileprivate static func listRepoFiles(repoID: String) async throws -> [HFTreeEntry] {
        let url = URL(
            string: "https://huggingface.co/api/models/\(repoID)/tree/main?recursive=true"
        )!
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.throwIfNonOK(response: response, url: url)

        struct RawEntry: Decodable {
            let type: String
            let path: String
            let size: Int?
        }
        let raw = try JSONDecoder().decode([RawEntry].self, from: data)
        return raw.compactMap { entry in
            guard entry.type == "file" else { return nil }
            return HFTreeEntry(path: entry.path, size: entry.size)
        }
    }

    /// Download a single file from `https://huggingface.co/{repo}/resolve/main/{path}`
    /// directly to disk. Follows redirects (HF returns 302 to CloudFront / xet).
    fileprivate static func downloadFile(
        repoID: String,
        path: String,
        to destination: URL
    ) async throws {
        let encodedPath = path.split(separator: "/")
            .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        let url = URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(encodedPath)")!

        // Stream to a tmp file first, then atomic-move on success.
        let (tempURL, response) = try await URLSession.shared.download(from: url)
        try Self.throwIfNonOK(response: response, url: url)

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    fileprivate static func throwIfNonOK(response: URLResponse, url: URL) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw WeightLoaderError.httpError(url, status: http.statusCode)
        }
    }
}
