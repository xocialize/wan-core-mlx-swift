// wan-granule-layout — CLI over WanCore's GranuleLayout writer (HV2).
// Lays a Wan expert safetensors checkpoint out as per-block contiguous granule
// files with 16 KiB-aligned tensor offsets + manifest.json, the on-disk format
// `BlockStreamer` streams from. Byte-copies only (quantization-agnostic);
// F_NOCACHE both ends so a 28.6 GB layout never flushes the page cache.
//
//   swift run -c release wan-granule-layout <model.safetensors> --out <dir>
//     [--block-prefix blocks.]
//
// A14B runs it once per expert (high_noise_model / low_noise_model → two dirs).

import Foundation
import WanCore

func argValue(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

let positional = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("--") }
let flagValues = Set(
    CommandLine.arguments.enumerated().compactMap { (i, a) in
        a.hasPrefix("--") && i + 1 < CommandLine.arguments.count
            ? CommandLine.arguments[i + 1] : nil
    })

guard let sourcePath = positional.first(where: { !flagValues.contains($0) }) else {
    print("usage: wan-granule-layout <model.safetensors> --out <dir> [--block-prefix blocks.]")
    exit(2)
}
let source = URL(filePath: sourcePath)
let outDir = URL(
    filePath: argValue("--out")
        ?? source.deletingPathExtension().path + ".granules")
let blockPrefix = argValue("--block-prefix") ?? "blocks."

print("layout: \(source.path)")
print("   out: \(outDir.path)")
do {
    let result = try WanGranuleLayout.write(
        safetensors: source, outputDir: outDir, blockPrefix: blockPrefix
    ) { done, total in
        if done % 8 == 0 || done == total {
            print("  block \(done)/\(total)")
            fflush(stdout)
        }
    }
    let m = result.manifest
    print(String(
        format: "done: %d blocks · %d tensors/block · %.2f GiB streamed "
            + "(%.2f GiB on disk incl. padding) · %.1fs (%.2f GiB/s) · %d global keys",
        m.blockCount, m.blocks[0].tensors.count,
        Double(m.streamedBytes) / 1_073_741_824,
        Double(result.writtenBytes) / 1_073_741_824,
        result.wallSeconds,
        Double(result.writtenBytes) / result.wallSeconds / 1_073_741_824,
        m.globalKeys.count))
} catch {
    print("FAILED: \(error.localizedDescription)")
    exit(1)
}
