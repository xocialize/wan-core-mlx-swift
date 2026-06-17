// WanProfiler — the shared, env-gated performance instrument for the whole Wan family.
//
// Generalizes the original `StageProfiler` (which only covered the vae22 streaming decode) into
// a substrate-resident instrument every weight-class consumer shares, so VACE-1.3B / Bernini-A14B /
// TI2V-5B / Helios / Phantom all emit the SAME CSV schema and are directly comparable.
//
// ── Design ──────────────────────────────────────────────────────────────────────────────────
//  • GLOBAL accessor (`WanProfiler.shared`), not a threaded parameter — the hot seams (WanModel
//    `runBlocks`, the per-consumer denoise loop) are deep in the call tree; threading a profiler
//    through every signature would be a large, invasive diff. The instrument is a process-wide
//    singleton whose hooks are NO-OPS unless `WAN_PROFILE` is set (zero cost when off).
//  • Two granularities (the original had only the distorting one):
//      - `region(...)`  COARSE — synchronize at the boundaries only, no forced intermediate `eval`.
//        Relies on the body's own `eval` (every hot Wan path self-evals per step / per chunk /
//        per block at large seqLen), so wall-clock is honest. This is the default — the phase
//        breakdown that proves where the time goes.
//      - `barrier(...)` FINE — forces `eval(result + materialize)` then synchronize (the original
//        `StageProfiler.stage` behavior). Serializes work that normally overlaps, so it OVERSTATES
//        wall-clock and is for RELATIVE attribution inside ONE opt-in phase (e.g. per-block inside
//        the DiT under `WAN_PROFILE_DEEP=blocks`). Never run deep on everything at once.
//  • Memory: records MLX `active`/`cache`/`peak` AND the OS `phys_footprint` (task_info). The
//    skill's load-bearing lesson — under the `Memory.cacheLimit` cap, `peakMemory` counts
//    cumulative allocations and MISLEADS (reads ~76 while phys is ~41); the governor's true basis
//    is `phys_footprint`. We capture both so the CSV shows the cap's effect and the real RAM.
//
// PROFILING BUILD ONLY in spirit, but safe to leave in: the hooks early-return when disabled.
// Env:
//   WAN_PROFILE=1            enable coarse region/step/phase timing + the CSV dump
//   WAN_PROFILE_DEEP=blocks  additionally enable the fine per-block barrier inside runBlocks
//                            (comma-separated list; future: attn, vae)

import Foundation
import Darwin
import MLX

/// OS `phys_footprint` (bytes) via `task_info(TASK_VM_INFO)` — the same number Activity Monitor's
/// "Memory" column and the engine's MemoryGovernor are grounded on. Returns 0 if the call fails.
/// This is the figure to declare `QuantFootprint.residentBytes` from, NOT `Memory.peakMemory`.
public func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

public final class WanProfiler: @unchecked Sendable {
    /// Process-wide instrument. Hooks no-op unless `enabled`.
    public static let shared = WanProfiler()

    public struct Row {
        public let group: String      // coarse bucket: "phase" / "denoise" / "blocks" / "vae" …
        public let label: String      // e.g. "text_encode", "step", "block", "decode"
        public let index: Int         // step / block / chunk index (-1 = whole)
        public let ms: Double
        public let activeGB: Double    // MLX active allocations
        public let cacheGB: Double     // MLX reclaimable buffer cache
        public let peakGB: Double      // MLX cumulative high-water (MISLEADS under the cap)
        public let physGB: Double      // OS phys_footprint (the truthful RAM figure)
        public let note: String        // free-form (seqLen, cfgBatch, dtype, …)
    }

    private let lock = NSLock()
    private var _rows: [Row] = []
    public let enabled: Bool
    private let deep: Set<String>

    public init() {
        let env = ProcessInfo.processInfo.environment
        self.enabled = (env["WAN_PROFILE"].map { $0 != "0" }) ?? false
        self.deep = Set((env["WAN_PROFILE_DEEP"] ?? "")
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty })
    }

    /// Is the fine per-`subsystem` drill-down requested? (`WAN_PROFILE_DEEP=blocks,attn`)
    public func deepEnabled(_ subsystem: String) -> Bool { enabled && deep.contains(subsystem) }

    public var rows: [Row] { lock.withLock { _rows } }
    public func reset() { lock.withLock { _rows.removeAll() } }

    private func gb(_ bytes: Int) -> Double { Double(bytes) / 1_073_741_824.0 }

    private func record(_ group: String, _ label: String, _ index: Int, _ ms: Double, _ note: String) {
        let s = Memory.snapshot()
        let row = Row(group: group, label: label, index: index, ms: ms,
                      activeGB: gb(s.activeMemory), cacheGB: gb(s.cacheMemory),
                      peakGB: gb(s.peakMemory),
                      physGB: Double(physFootprintBytes()) / 1_073_741_824.0, note: note)
        lock.withLock { _rows.append(row) }
        // Stream each row to stdout (CSV column order, `[WANPROF]`-tagged) so the live test app —
        // which never calls `dumpCSV()` — still captures the data for free off its serve path.
        // `grep '^\[WANPROF\]' | cut -d' ' -f2-` reconstructs the CSV. The CLI adds the rollup.
        print("[WANPROF] \(row.group),\(row.label),\(row.index),\(f(row.ms)),\(f3(row.activeGB)),"
            + "\(f3(row.cacheGB)),\(f3(row.peakGB)),\(f3(row.physGB)),\(row.note)")
    }

    // MARK: - COARSE region (the honest one)

    /// Time a region — NO forced intermediate `eval`. Accurate when the body self-`eval`s (every
    /// hot Wan path does: per-step `eval(latents)`, per-chunk VAE `eval`, per-block `eval` at large
    /// seqLen). `eval` is the synchronous barrier in mlx-swift (it blocks until the GPU work
    /// completes), so the body's own `eval` bounds the timed interval — no extra synchronize needed.
    /// The prior region/step already `eval`'d and blocked, so t0 starts on a drained queue.
    @discardableResult
    public func region<T>(_ group: String, _ label: String, index: Int = -1, note: String = "",
                          _ body: () throws -> T) rethrows -> T {
        guard enabled else { return try body() }
        let t0 = DispatchTime.now()
        let result = try body()
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000.0
        record(group, label, index, ms, note)
        return result
    }

    // MARK: - FINE barrier (the distorting one — opt-in, single phase)

    /// Time a stage, forcing `eval` of the result (+ any extra arrays the stage materialized, e.g.
    /// cache slots) before stopping the clock. Serializes work — OVERSTATES wall-clock; use only
    /// for relative attribution inside one opt-in phase.
    @discardableResult
    public func barrier<T>(_ group: String, _ label: String, index: Int = -1, note: String = "",
                           materialize extra: [MLXArray] = [], _ body: () throws -> T) rethrows -> T {
        guard enabled else { return try body() }
        let t0 = DispatchTime.now()
        let result = try body()
        // `eval` blocks until the GPU work finishes — the synchronization point for this stage.
        if let arr = result as? MLXArray { eval([arr] + extra) } else { eval(extra) }
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000.0
        record(group, label, index, ms, note)
        return result
    }

    /// Record current memory at a point in time without timing anything (phase boundary marker).
    public func mark(_ group: String, _ label: String, index: Int = -1, note: String = "") {
        guard enabled else { return }
        record(group, label, index, 0, note)
    }

    // MARK: - Reporting

    /// Raw CSV + rollups to stdout. `denominators` lets the CLI print the headline normalized
    /// metrics (ms/forward, ms/frame, ms/1k-token) that make weight classes comparable.
    public func dumpCSV(denominators: [String: Double] = [:]) {
        let rows = self.rows
        print("group,label,index,ms,active_gb,cache_gb,peak_gb,phys_gb,note")
        for r in rows {
            print("\(r.group),\(r.label),\(r.index),\(f(r.ms)),\(f3(r.activeGB)),\(f3(r.cacheGB)),"
                + "\(f3(r.peakGB)),\(f3(r.physGB)),\(r.note)")
        }

        // Per-(group/label) rollup: total ms, count, mean — the headline attribution.
        var totals: [String: (ms: Double, n: Int)] = [:]
        for r in rows {
            let k = "\(r.group)/\(r.label)"
            totals[k, default: (0, 0)] = (totals[k]!.ms + r.ms, totals[k]!.n + 1)
        }
        print("\n--- per (group/label): total_ms, count, mean_ms (sorted by total) ---")
        for (k, v) in totals.sorted(by: { $0.value.ms > $1.value.ms }) {
            print("\(k),\(f(v.ms)),\(v.n),\(f(v.ms / Double(max(v.n, 1))))")
        }

        let grand = rows.reduce(0.0) { $0 + $1.ms }
        let peakPhys = rows.map(\.physGB).max() ?? 0
        print("\n--- headline ---")
        print("total_timed_ms,\(f(grand))")
        print("peak_phys_gb,\(f3(peakPhys))")
        for (name, d) in denominators.sorted(by: { $0.key < $1.key }) where d > 0 {
            print("ms_per_\(name),\(f(grand / d))")
        }
    }

    private func f(_ x: Double) -> String { String(format: "%.2f", x) }
    private func f3(_ x: Double) -> String { String(format: "%.3f", x) }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T { lock(); defer { unlock() }; return body() }
}
