// HV2 weight streaming, part 1: the granule layout (NEUROSTREAM-ACTIONS HV2;
// prototype receipt probes/hv2_stream_proto.out, seam probe probes/pc_mlx_seam.out).
//
// A "granule" is one transformer block's full parameter set serialized as a single
// contiguous file with every tensor offset 16 KiB-aligned — the unit the
// `BlockStreamer` preads into a live slot between group boundaries. This file owns:
//   - minimal safetensors header parsing (offsets only; no MLX, no mmap),
//   - the layout writer (safetensors → per-block granule files + manifest.json),
//   - the manifest types the streamer binds against.
//
// The writer copies BYTES — no dtype conversion, so it is quantization-agnostic
// (bf16 blocks are 27 tensors, int4 blocks are 47 incl. packed weight/scales/biases).
// Both source reads and granule writes run F_NOCACHE: a 28.6 GB expert must not
// flush the UBC on layout (and honest-cold streaming receipts depend on granules
// never entering the page cache — the prototype's cold-number recipe).

import Foundation

public enum GranuleLayoutError: LocalizedError {
    case io(String)
    case badSafetensors(String)
    case badManifest(String)

    public var errorDescription: String? {
        switch self {
        case .io(let m): return "granule layout IO error: \(m)"
        case .badSafetensors(let m): return "bad safetensors: \(m)"
        case .badManifest(let m): return "bad granule manifest: \(m)"
        }
    }
}

// MARK: - Safetensors header (offsets-only parse)

/// One tensor entry from a safetensors header: dtype string, shape, and the byte
/// range within the file's data section (absolute offsets after `dataStart`).
public struct SafetensorsTensorEntry: Sendable {
    public let name: String
    public let dtype: String  // safetensors dtype string ("BF16", "F32", "U32", …)
    public let shape: [Int]
    public let start: Int  // relative to dataStart
    public let end: Int

    public var nbytes: Int { end - start }
}

/// Minimal safetensors header reader: 8-byte LE header length + JSON dict.
/// Reads only the header — never the tensor data (a 28.6 GB expert parses in ms).
public struct SafetensorsHeader: Sendable {
    public let tensors: [String: SafetensorsTensorEntry]
    public let dataStart: Int  // absolute file offset of the data section

    public static func read(url: URL) throws -> SafetensorsHeader {
        guard let fh = FileHandle(forReadingAtPath: url.path) else {
            throw GranuleLayoutError.io("cannot open \(url.path)")
        }
        defer { try? fh.close() }
        guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
            throw GranuleLayoutError.badSafetensors("short read on header length")
        }
        let headerLen = lenData.withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self)) }
        guard headerLen > 0, headerLen < 512 << 20 else {
            throw GranuleLayoutError.badSafetensors("implausible header length \(headerLen)")
        }
        guard let headerData = try fh.read(upToCount: headerLen), headerData.count == headerLen
        else {
            throw GranuleLayoutError.badSafetensors("short read on header body")
        }
        guard
            let obj = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            throw GranuleLayoutError.badSafetensors("header is not a JSON object")
        }
        var tensors: [String: SafetensorsTensorEntry] = [:]
        for (name, value) in obj {
            if name == "__metadata__" { continue }
            guard let dict = value as? [String: Any],
                let dtype = dict["dtype"] as? String,
                let shape = dict["shape"] as? [Int],
                let offsets = dict["data_offsets"] as? [Int], offsets.count == 2
            else {
                throw GranuleLayoutError.badSafetensors("malformed entry for \(name)")
            }
            tensors[name] = SafetensorsTensorEntry(
                name: name, dtype: dtype, shape: shape, start: offsets[0], end: offsets[1])
        }
        return SafetensorsHeader(tensors: tensors, dataStart: 8 + headerLen)
    }
}

// MARK: - Manifest

/// Per-tensor record inside one block granule. `key` is the parameter path
/// RELATIVE to the block (e.g. `self_attn.q.weight`) so the streamer can map it
/// onto any block index.
public struct GranuleTensor: Codable, Sendable, Equatable {
    public let key: String
    public let dtype: String  // safetensors dtype string, preserved verbatim
    public let shape: [Int]
    public let offset: Int  // within the granule file; 16 KiB-aligned
    public let nbytes: Int
}

public struct GranuleBlock: Codable, Sendable {
    public let index: Int
    public let file: String  // relative to the manifest's directory
    public let bytes: Int  // granule file size (offsets + padding included)
    public let tensors: [GranuleTensor]

    enum CodingKeys: String, CodingKey {
        case index, file, bytes, tensors
    }
}

/// The granule directory manifest — everything the `BlockStreamer` needs to bind
/// slots and refill them, without touching the source safetensors (which remains
/// the source for the small resident "global" tensors).
public struct GranuleManifest: Codable, Sendable {
    public let version: Int
    public let alignment: Int
    public let sourceFile: String
    public let sourceSize: Int
    public let blockPrefix: String
    public let blockCount: Int
    /// Sum of raw tensor bytes across all blocks (the streamed bytes per sweep —
    /// the `B·F/(2·S)` numerator side of the runtime gate).
    public let streamedBytes: Int
    public let blocks: [GranuleBlock]
    /// Keys in the source safetensors that are NOT per-block (embeddings, head, …)
    /// — loaded resident by the consumer; may include conversion artifacts the
    /// model never loads (e.g. the int4 checkpoints' serialized `freqs`).
    public let globalKeys: [String]

    enum CodingKeys: String, CodingKey {
        case version
        case alignment
        case sourceFile = "source_file"
        case sourceSize = "source_size"
        case blockPrefix = "block_prefix"
        case blockCount = "block_count"
        case streamedBytes = "streamed_bytes"
        case blocks
        case globalKeys = "global_keys"
    }

    public static let currentVersion = 1
    public static let manifestFileName = "manifest.json"

    public static func load(from dir: URL) throws -> GranuleManifest {
        let url = dir.appendingPathComponent(manifestFileName)
        let manifest: GranuleManifest
        do {
            manifest = try JSONDecoder().decode(
                GranuleManifest.self, from: Data(contentsOf: url))
        } catch {
            throw GranuleLayoutError.badManifest(
                "\(url.path): \(error.localizedDescription)")
        }
        guard manifest.version == currentVersion else {
            throw GranuleLayoutError.badManifest(
                "version \(manifest.version) != \(currentVersion)")
        }
        guard manifest.blocks.count == manifest.blockCount else {
            throw GranuleLayoutError.badManifest(
                "block count \(manifest.blocks.count) != declared \(manifest.blockCount)")
        }
        // The streamer requires a uniform tensor table across blocks (slot arrays
        // are allocated once and reused for every block of that local index).
        let reference = manifest.blocks[0].tensors.map { "\($0.key)|\($0.dtype)|\($0.shape)" }
        for block in manifest.blocks.dropFirst() {
            let table = block.tensors.map { "\($0.key)|\($0.dtype)|\($0.shape)" }
            guard table == reference else {
                throw GranuleLayoutError.badManifest(
                    "block \(block.index) tensor table differs from block 0 — "
                        + "granule streaming requires uniform blocks")
            }
        }
        return manifest
    }
}

// MARK: - Raw file IO helpers (shared with the streamer)

enum GranuleIO {
    static let alignment = 16384

    static func openRead(_ path: String, noCache: Bool = true) throws -> Int32 {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw GranuleLayoutError.io("open(\(path)) errno \(errno)") }
        if noCache { _ = fcntl(fd, F_NOCACHE, 1) }
        return fd
    }

    static func preadFull(fd: Int32, into dst: UnsafeMutableRawPointer, count: Int, offset: Int)
        throws
    {
        var done = 0
        while done < count {
            let n = pread(fd, dst + done, min(32 << 20, count - done), off_t(offset + done))
            guard n > 0 else {
                throw GranuleLayoutError.io("pread failed at offset \(offset + done): errno \(errno)")
            }
            done += n
        }
    }

    static func writeFull(fd: Int32, from src: UnsafeRawPointer, count: Int) throws {
        var done = 0
        while done < count {
            let n = write(fd, src + done, min(32 << 20, count - done))
            guard n > 0 else { throw GranuleLayoutError.io("write failed: errno \(errno)") }
            done += n
        }
    }
}

// MARK: - Layout writer

public enum GranuleLayout {

    /// Canonical within-block tensor order: WanAttentionBlock forward order, so a
    /// granule read visits tensors roughly as compute will. Suffix-prefix ranked;
    /// unknown suffixes sort after the known ones (robustness across Wan variants),
    /// lexicographic within a rank. Order affects locality only — correctness never
    /// depends on it (a slot is fully refilled before compute touches any of it).
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

    static func forwardRank(_ key: String) -> Int {
        for (i, p) in forwardOrderPrefixes.enumerated() where key.hasPrefix(p) { return i }
        return forwardOrderPrefixes.count
    }

    public struct Result: Sendable {
        public let manifest: GranuleManifest
        public let outputDir: URL
        public let wallSeconds: Double
        public let writtenBytes: Int
    }

    /// Lay a safetensors checkpoint out as per-block granule files + manifest.
    ///
    /// - Parameters:
    ///   - safetensors: source checkpoint (e.g. `high_noise_model.safetensors`).
    ///   - outputDir: created if needed; receives `block_NNN.granule` + `manifest.json`.
    ///   - blockPrefix: parameter-path prefix that marks per-block tensors.
    ///   - progress: optional per-block callback (blockIndex, blockCount).
    @discardableResult
    public static func write(
        safetensors: URL,
        outputDir: URL,
        blockPrefix: String = "blocks.",
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> Result {
        let t0 = Date()
        let header = try SafetensorsHeader.read(url: safetensors)

        // Partition keys: `blocks.<i>.<suffix>` vs global.
        var perBlock: [Int: [(suffix: String, entry: SafetensorsTensorEntry)]] = [:]
        var globalKeys: [String] = []
        for (name, entry) in header.tensors {
            guard name.hasPrefix(blockPrefix),
                let dot = name.dropFirst(blockPrefix.count).firstIndex(of: "."),
                let index = Int(name.dropFirst(blockPrefix.count)[..<dot])
            else {
                globalKeys.append(name)
                continue
            }
            let suffix = String(name[name.index(after: dot)...])
            perBlock[index, default: []].append((suffix, entry))
        }
        guard !perBlock.isEmpty else {
            throw GranuleLayoutError.badSafetensors(
                "no keys under block prefix '\(blockPrefix)'")
        }
        let blockCount = perBlock.keys.max()! + 1
        guard perBlock.count == blockCount else {
            throw GranuleLayoutError.badSafetensors(
                "non-contiguous block indices (have \(perBlock.count), max index \(blockCount - 1))")
        }

        try FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true)

        let srcFd = try GranuleIO.openRead(safetensors.path)
        defer { close(srcFd) }

        // Page-aligned bounce buffer for F_NOCACHE-friendly copies.
        let bufBytes = 32 << 20
        var bufRaw: UnsafeMutableRawPointer? = nil
        guard posix_memalign(&bufRaw, GranuleIO.alignment, bufBytes) == 0, let buf = bufRaw
        else { throw GranuleLayoutError.io("posix_memalign failed") }
        defer { free(buf) }

        var blocks: [GranuleBlock] = []
        var streamedBytes = 0
        var writtenBytes = 0
        let align = GranuleIO.alignment

        for b in 0..<blockCount {
            let ordered = perBlock[b]!.sorted {
                (forwardRank($0.suffix), $0.suffix) < (forwardRank($1.suffix), $1.suffix)
            }
            let fileName = String(format: "block_%03d.granule", b)
            let dstPath = outputDir.appendingPathComponent(fileName).path
            let dstFd = open(dstPath, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
            guard dstFd >= 0 else {
                throw GranuleLayoutError.io("create \(dstPath) errno \(errno)")
            }
            _ = fcntl(dstFd, F_NOCACHE, 1)

            var tensors: [GranuleTensor] = []
            var cursor = 0
            for (suffix, entry) in ordered {
                // Copy tensor bytes, then zero-pad the file to the next 16 KiB
                // boundary so the NEXT tensor's offset is aligned.
                var remaining = entry.nbytes
                var srcOff = header.dataStart + entry.start
                while remaining > 0 {
                    let n = min(bufBytes, remaining)
                    try GranuleIO.preadFull(fd: srcFd, into: buf, count: n, offset: srcOff)
                    try GranuleIO.writeFull(fd: dstFd, from: buf, count: n)
                    srcOff += n
                    remaining -= n
                }
                tensors.append(
                    GranuleTensor(
                        key: suffix, dtype: entry.dtype, shape: entry.shape,
                        offset: cursor, nbytes: entry.nbytes))
                streamedBytes += entry.nbytes
                cursor += entry.nbytes
                let padded = (cursor + align - 1) / align * align
                if padded > cursor {
                    memset(buf, 0, padded - cursor)
                    try GranuleIO.writeFull(fd: dstFd, from: buf, count: padded - cursor)
                    cursor = padded
                }
            }
            close(dstFd)
            writtenBytes += cursor
            blocks.append(
                GranuleBlock(index: b, file: fileName, bytes: cursor, tensors: tensors))
            progress?(b + 1, blockCount)
        }

        let sourceSize =
            (try? FileManager.default.attributesOfItem(atPath: safetensors.path)[.size] as? Int)
            .flatMap { $0 } ?? 0
        let manifest = GranuleManifest(
            version: GranuleManifest.currentVersion,
            alignment: align,
            sourceFile: safetensors.lastPathComponent,
            sourceSize: sourceSize,
            blockPrefix: blockPrefix,
            blockCount: blockCount,
            streamedBytes: streamedBytes,
            blocks: blocks,
            globalKeys: globalKeys.sorted())
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(
            to: outputDir.appendingPathComponent(GranuleManifest.manifestFileName))

        // Re-validate through the reader so a written layout is a loadable layout.
        let reloaded = try GranuleManifest.load(from: outputDir)
        return Result(
            manifest: reloaded, outputDir: outputDir,
            wallSeconds: Date().timeIntervalSince(t0), writtenBytes: writtenBytes)
    }
}
