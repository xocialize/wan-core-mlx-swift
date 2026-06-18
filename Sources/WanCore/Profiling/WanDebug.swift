// WanDebug — env-gated array-stat tracing for the whole Wan family. Distinct from WanProfiler
// (timing/memory): this prints VALUE stats of the hot tensors so the testing side can localize
// where an output goes black / NaN — a diverging denoise latent vs a zeroing encode vs decode —
// WITHOUT the Python oracle. Off by default; enable with `WAN_DEBUG_STATS=1`. Every stat forces a
// sync (it materializes the array), so it is debug-only — the guard makes it a true no-op otherwise.

import Foundation
import MLX

public enum WanDebug {
    /// True when `WAN_DEBUG_STATS` is set (to anything non-empty).
    public static let statsEnabled: Bool = {
        if let v = ProcessInfo.processInfo.environment["WAN_DEBUG_STATS"], !v.isEmpty { return true }
        return false
    }()

    /// Print value stats of `x` under `label` — shape, L2 norm, min, max, mean, NaN/Inf counts, and
    /// an all-zero flag (the black-output tell). No-op unless `WAN_DEBUG_STATS` is set.
    /// `[WANSTATS]`-tagged so it greps cleanly alongside `[WANPROF]`.
    public static func stats(_ label: String, _ x: MLXArray) {
        guard statsEnabled else { return }
        let xf = x.asType(.float32)
        // NaN != NaN; Inf detected via |x| == Inf. Counts computed before min/max (which NaN-propagate).
        let nan = (xf .!= xf).asType(.int32).sum().item(Int32.self)
        let inf = (MLX.abs(xf) .== Float.infinity).asType(.int32).sum().item(Int32.self)
        let norm = MLX.sqrt(xf.square().sum()).item(Float.self)
        let mn = xf.min().item(Float.self)
        let mx = xf.max().item(Float.self)
        let mean = xf.mean().item(Float.self)
        let allZero = (mn == 0 && mx == 0) ? " ALL-ZERO" : ""
        print("[WANSTATS] \(label) shape=\(x.shape) norm=\(fmt(norm)) "
            + "min=\(fmt(mn)) max=\(fmt(mx)) mean=\(fmt(mean)) nan=\(nan) inf=\(inf)\(allZero)")
    }

    private static func fmt(_ v: Float) -> String { String(format: "%.4g", v) }
}
