// Prompt → UMT5 features, mirroring mlx_video/models/wan_2/utils.py
// (_clean_text + encode_text @ 87db56a). Tokenizer = google/umt5-xxl, same
// repo mlx-video loads (the published Bernini weight repos ship no tokenizer
// files).

import Foundation
import MLX
import Tokenizers

/// Clean text matching the official Wan2.2 tokenizer preprocessing: double
/// HTML unescape + whitespace normalization. (Upstream also applies
/// ftfy.fix_text when installed — a mojibake repair that is a no-op on
/// well-formed UTF-8 input; not ported.)
func cleanText(_ text: String) -> String {
    var t = text
    for _ in 0..<2 {
        t = CFXMLCreateStringByUnescapingEntities(nil, t as CFString, nil) as String
    }
    t = t.replacingOccurrences(
        of: "\\s+", with: " ", options: .regularExpression)
    return t.trimmingCharacters(in: .whitespacesAndNewlines)
}

/// The HF repo the umT5 tokenizer loads from (mlx-video parity).
public let umt5TokenizerRepo = "google/umt5-xxl"

/// T5 pad token id.
private let t5PadID = 0

/// Encode a text prompt with the UMT5 encoder.
/// Mirrors `encode_text`: pad to `textLen` with the pad id + attention mask,
/// run the encoder, return only the non-padding tokens — [L, dim].
public func encodeText(
    encoder: UMT5EncoderModel,
    tokenizer: any Tokenizer,
    prompt: String,
    textLen: Int = 512
) -> MLXArray {
    var ids = tokenizer.encode(text: cleanText(prompt))
    if ids.count > textLen {
        // HF truncation keeps the trailing </s> special token
        ids = Array(ids.prefix(textLen - 1)) + [ids.last!]
    }
    let seqLen = ids.count
    let padded = ids + Array(repeating: t5PadID, count: textLen - seqLen)
    let mask = Array(repeating: Int32(1), count: seqLen)
        + Array(repeating: Int32(0), count: textLen - seqLen)

    let idsArr = MLXArray(padded.map(Int32.init), [1, textLen])
    let maskArr = MLXArray(mask, [1, textLen])

    let embeddings = encoder(idsArr, mask: maskArr)
    return embeddings[0, ..<seqLen]
}
