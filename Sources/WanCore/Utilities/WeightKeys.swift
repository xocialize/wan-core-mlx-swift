// BerniniWeightKeys — the pinned safetensors key contract for the published
// checkpoints (mlx-community/Bernini-R-{bf16,int4}; same layout as the local
// mirrors). Verified against the actual headers 2026-06-12: DiT 1095 keys/expert
// (bf16) / 1896 (int4, +scales/biases on the 10 block Linears, +`freqs`);
// t5_encoder 242; vae 194. Loaders must refuse partial loads (0 missing /
// 0 unused) against these sets — drift here is how silent zero-weights ship.

import Foundation

public enum BerniniWeightKeys {

    /// Quantized Linear paths within one DiT block (the int4 recipe's scope:
    /// attention q/k/v/o + FFN; norms / modulation / embeddings / head stay bf16).
    static let blockLinearPaths = [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        "ffn.fc1", "ffn.fc2",
    ]

    /// Expert (high_noise_model / low_noise_model) key set.
    public static func ditKeys(layers: Int = 40, quantized: Bool = false) -> Set<String> {
        var keys = Set<String>()
        for i in 0..<layers {
            let b = "blocks.\(i)"
            for path in blockLinearPaths {
                keys.insert("\(b).\(path).weight")
                keys.insert("\(b).\(path).bias")
                if quantized {
                    keys.insert("\(b).\(path).scales")
                    keys.insert("\(b).\(path).biases")
                }
            }
            for attn in ["self_attn", "cross_attn"] {
                keys.insert("\(b).\(attn).norm_q.weight")
                keys.insert("\(b).\(attn).norm_k.weight")
            }
            keys.insert("\(b).modulation")
            keys.insert("\(b).norm3.weight")
            keys.insert("\(b).norm3.bias")
        }
        for g in [
            "patch_embedding_proj", "text_embedding_0", "text_embedding_1",
            "time_embedding_0", "time_embedding_1", "time_projection", "head.head",
        ] {
            keys.insert("\(g).weight")
            keys.insert("\(g).bias")
        }
        keys.insert("head.modulation")
        if quantized {
            // The quantize script serializes the model's precomputed rope table
            // ([1024, head_dim/2, 2] fp32); loaders accept-and-ignore it.
            keys.insert("freqs")
        }
        return keys
    }

    /// t5_encoder.safetensors key set (UMT5-xxl: per-block relative position bias).
    public static func t5Keys(layers: Int = 24) -> Set<String> {
        var keys = Set<String>()
        for i in 0..<layers {
            let b = "blocks.\(i)"
            for path in ["attn.q", "attn.k", "attn.v", "attn.o",
                         "ffn.fc1", "ffn.fc2", "ffn.gate_proj",
                         "norm1", "norm2"] {
                keys.insert("\(b).\(path).weight")
            }
            keys.insert("\(b).pos_embedding.embedding.weight")
        }
        keys.insert("token_embedding.weight")
        keys.insert("norm.weight")
        return keys
    }
}
