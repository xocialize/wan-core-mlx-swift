// 1:1 translation of mlx_video/models/wan_2/scheduler.py (mlx-video @ 87db56a).
// Flow matching schedulers: Euler, DPM++(2M), UniPC (bh2) — UniPC is the
// official Wan2.2 default and what Bernini-R uses (shift 3.0, 40 steps).
// Scalar lambda/coefficient math runs in Double (numpy float64 parity); the
// order>=2 coefficient systems are solved with float64 Gaussian elimination.

import Foundation
import MLX

/// Compute the shifted sigma schedule matching the official Wan2.2 scheduler.
/// Returns numSteps+1 values (the last being 0.0 for the terminal state).
func computeSigmas(
    numSteps: Int, shift: Double = 1.0, numTrainTimesteps: Int = 1000
) -> [Float] {
    // sigma bounds from the unshifted training schedule (constructor uses shift=1):
    // alphas = linspace(1, 1/N, N) reversed; sigmas = 1 - alphas
    // sigma_max = (N-1)/N, sigma_min = 0.0
    let n = Double(numTrainTimesteps)
    let sigmaMax = 1.0 - 1.0 / n
    let sigmaMin = 0.0

    // Interpolate, then apply shift once (matching set_timesteps)
    var sigmas: [Float] = []
    for i in 0..<numSteps {
        let frac = Double(i) / Double(numSteps)  // linspace over numSteps+1, dropping last
        let sigma = sigmaMax + (sigmaMin - sigmaMax) * frac
        let shifted = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        sigmas.append(Float(shifted))
    }
    sigmas.append(0.0)
    return sigmas
}

/// log-SNR: lambda(sigma) = log((1-sigma)/sigma); ±inf at the boundaries
/// (matching torch.log behavior in the official code).
private func lambdaOf(_ sigma: Double) -> Double {
    if sigma >= 1.0 { return -.infinity }
    if sigma <= 0.0 { return .infinity }
    return log((1.0 - sigma) / sigma)
}

/// Solve Ax = b in float64 via Gaussian elimination with partial pivoting
/// (stands in for np.linalg.solve on the small UniPC coefficient systems).
func solveLinearF64(_ a: [[Double]], _ b: [Double]) -> [Double] {
    let n = b.count
    var m = a
    var v = b
    for col in 0..<n {
        var pivot = col
        for row in (col + 1)..<n where abs(m[row][col]) > abs(m[pivot][col]) {
            pivot = row
        }
        if pivot != col {
            m.swapAt(col, pivot)
            v.swapAt(col, pivot)
        }
        for row in (col + 1)..<n {
            let factor = m[row][col] / m[col][col]
            for j in col..<n {
                m[row][j] -= factor * m[col][j]
            }
            v[row] -= factor * v[col]
        }
    }
    var x = [Double](repeating: 0, count: n)
    for row in stride(from: n - 1, through: 0, by: -1) {
        var sum = v[row]
        for j in (row + 1)..<n {
            sum -= m[row][j] * x[j]
        }
        x[row] = sum / m[row][row]
    }
    return x
}

/// 1st-order Euler scheduler for flow matching diffusion.
public final class FlowMatchEulerScheduler {
    public let numTrainTimesteps: Int
    public private(set) var timesteps: [Float] = []
    public private(set) var sigmas: [Float] = []
    private var stepIndex = 0

    public init(numTrainTimesteps: Int = 1000) {
        self.numTrainTimesteps = numTrainTimesteps
    }

    public func setTimesteps(_ numSteps: Int, shift: Double = 1.0) {
        sigmas = computeSigmas(
            numSteps: numSteps, shift: shift, numTrainTimesteps: numTrainTimesteps)
        // Integer timesteps to match reference (model trained with int timesteps)
        timesteps = sigmas.dropLast().map { Float(Int64($0 * Float(numTrainTimesteps))) }
        stepIndex = 0
    }

    /// Euler step: x_next = x + (sigma_next - sigma_cur) * v.
    public func step(
        modelOutput: MLXArray, timestep: Float, sample: MLXArray
    ) -> MLXArray {
        let dt = sigmas[stepIndex + 1] - sigmas[stepIndex]
        let xNext = sample + dt * modelOutput
        stepIndex += 1
        return xNext
    }

    public func reset() { stepIndex = 0 }
}

/// DPM-Solver++(2M) for flow matching diffusion.
public final class FlowDPMPP2MScheduler {
    public let numTrainTimesteps: Int
    public let lowerOrderFinal: Bool
    public private(set) var timesteps: [Float] = []
    public private(set) var sigmas: [Float] = []
    private var stepIndex = 0
    private var numSteps = 0
    private var prevX0: MLXArray?

    public init(numTrainTimesteps: Int = 1000, lowerOrderFinal: Bool = true) {
        self.numTrainTimesteps = numTrainTimesteps
        self.lowerOrderFinal = lowerOrderFinal
    }

    public func setTimesteps(_ numSteps: Int, shift: Double = 1.0) {
        sigmas = computeSigmas(
            numSteps: numSteps, shift: shift, numTrainTimesteps: numTrainTimesteps)
        timesteps = sigmas.dropLast().map { Float(Int64($0 * Float(numTrainTimesteps))) }
        stepIndex = 0
        self.numSteps = numSteps
        prevX0 = nil
    }

    public func step(
        modelOutput: MLXArray, timestep: Float, sample: MLXArray
    ) -> MLXArray {
        let i = stepIndex
        let sigmaCur = Double(sigmas[i])
        let sigmaNext = Double(sigmas[i + 1])

        // Convert velocity -> x0 prediction: x0 = sample - sigma * v
        let x0 = sample - Float(sigmaCur) * modelOutput

        // 1st order on first step, last step (if lowerOrderFinal and few steps)
        let useFirstOrder = prevX0 == nil
            || (lowerOrderFinal && i == numSteps - 1 && numSteps < 15)

        var xNext: MLXArray
        if useFirstOrder || sigmaNext == 0.0 {
            if sigmaNext == 0.0 {
                xNext = x0
            } else {
                let h = lambdaOf(sigmaNext) - lambdaOf(sigmaCur)
                let alphaNext = 1.0 - sigmaNext
                let coeffX = sigmaNext / sigmaCur
                let coeffX0 = alphaNext * expm1(-h)
                xNext = Float(coeffX) * sample - Float(coeffX0) * x0
            }
        } else {
            // 2nd order DPM++(2M) with midpoint correction
            let sigmaPrev = Double(sigmas[i - 1])
            let h = lambdaOf(sigmaNext) - lambdaOf(sigmaCur)
            let h0 = lambdaOf(sigmaCur) - lambdaOf(sigmaPrev)
            let r0 = h0 / h

            let d0 = x0
            let d1 = Float(1.0 / r0) * (x0 - prevX0!)

            let alphaNext = 1.0 - sigmaNext
            let expNegHM1 = expm1(-h)

            xNext = Float(sigmaNext / sigmaCur) * sample
                - Float(alphaNext * expNegHM1) * d0
                - Float(0.5 * alphaNext * expNegHM1) * d1
        }

        prevX0 = x0
        stepIndex += 1
        return xNext
    }

    public func reset() {
        stepIndex = 0
        prevX0 = nil
    }
}

/// UniPC (Unified Predictor-Corrector) for flow matching diffusion — the
/// official Wan2.2 default (bh2 basis).
public final class FlowUniPCScheduler {
    public let numTrainTimesteps: Int
    public let solverOrder: Int
    public let lowerOrderFinal: Bool
    public let disableCorrector: Set<Int>
    private let useCorrector: Bool

    public private(set) var timesteps: [Float] = []
    public private(set) var sigmas: [Float] = []
    private var stepIndex = 0
    private var numSteps = 0
    private var lowerOrderNums = 0
    /// Model output (x0) history for multi-step, stored newest-last.
    private var modelOutputs: [MLXArray?] = []
    private var lastSample: MLXArray?
    private var thisOrder = 1

    public init(
        numTrainTimesteps: Int = 1000,
        solverOrder: Int = 2,
        lowerOrderFinal: Bool = true,
        disableCorrector: [Int]? = nil,
        useCorrector: Bool = true
    ) {
        self.numTrainTimesteps = numTrainTimesteps
        self.solverOrder = solverOrder
        self.lowerOrderFinal = lowerOrderFinal
        self.useCorrector = useCorrector
        self.disableCorrector = Set(disableCorrector ?? [])
    }

    public func setTimesteps(_ numSteps: Int, shift: Double = 1.0) {
        sigmas = computeSigmas(
            numSteps: numSteps, shift: shift, numTrainTimesteps: numTrainTimesteps)
        timesteps = sigmas.dropLast().map { Float(Int64($0 * Float(numTrainTimesteps))) }
        stepIndex = 0
        self.numSteps = numSteps
        lowerOrderNums = 0
        modelOutputs = Array(repeating: nil, count: solverOrder)
        lastSample = nil
        thisOrder = 1
    }

    /// Convert velocity prediction to x0: x0 = sample - sigma * v.
    private func convertOutput(_ velocity: MLXArray, sample: MLXArray) -> MLXArray {
        sample - sigmas[stepIndex] * velocity
    }

    /// UniP predictor with B(h)=expm1(-h) basis (bh2 variant).
    private func uniPBh2(x0: MLXArray, sample: MLXArray, order: Int) -> MLXArray {
        let i = stepIndex
        let sigmaS0 = Double(sigmas[i])
        let sigmaT = Double(sigmas[i + 1])

        if sigmaT == 0.0 { return x0 }

        let h = lambdaOf(sigmaT) - lambdaOf(sigmaS0)
        let hh = -h  // negated for predict_x0

        let alphaT = 1.0 - sigmaT
        let hPhi1 = expm1(hh)
        let bH = hPhi1

        let m0 = modelOutputs.last!!
        // Base prediction
        var xT = Float(sigmaT / sigmaS0) * sample - Float(alphaT * hPhi1) * m0

        if order >= 2 {
            var rks: [Double] = []
            var d1s: [MLXArray] = []
            for k in 1..<order {
                let siIdx = i - k
                guard siIdx >= 0,
                      let mk = modelOutputs[modelOutputs.count - (k + 1)] else { break }
                let lambdaSk = lambdaOf(Double(sigmas[siIdx]))
                let rk = (lambdaSk - lambdaOf(sigmaS0)) / h
                if rk.isInfinite { break }
                rks.append(rk)
                d1s.append((mk - m0) / Float(rk))
            }

            if !d1s.isEmpty {
                let effectiveOrder = d1s.count + 1
                let rhosP: [Double]
                if effectiveOrder <= 2 {
                    // Analytic solution for order 2
                    rhosP = [0.5]
                } else {
                    var hPhiK = hPhi1 / hh - 1.0
                    var factorialI = 1.0
                    var rRows: [[Double]] = []
                    var bVals: [Double] = []
                    for j in 1..<effectiveOrder {
                        rRows.append(rks.map { pow($0, Double(j - 1)) })
                        bVals.append(hPhiK * factorialI / bH)
                        factorialI *= Double(j + 1)
                        hPhiK = hPhiK / hh - 1.0 / factorialI
                    }
                    rhosP = solveLinearF64(rRows, bVals)
                }

                var predRes = MLXArray.zeros(like: d1s[0])
                for (rho, d1) in zip(rhosP, d1s) {
                    predRes = predRes + Float(rho) * d1
                }
                xT = xT - Float(alphaT * bH) * predRes
            }
        }

        return xT
    }

    /// UniC corrector with B(h)=expm1(-h) basis (bh2 variant).
    private func uniCBh2(
        modelX0: MLXArray, lastSample: MLXArray, thisSample: MLXArray, order: Int
    ) -> MLXArray {
        let i = stepIndex
        let sigmaS0 = Double(sigmas[i - 1])
        let sigmaT = Double(sigmas[i])

        if sigmaT == 0.0 { return thisSample }

        let h = lambdaOf(sigmaT) - lambdaOf(sigmaS0)
        let hh = -h  // negated for predict_x0

        let alphaT = 1.0 - sigmaT
        let hPhi1 = expm1(hh)
        let bH = hPhi1

        let m0 = modelOutputs.last!!
        // Re-derive base from last_sample
        let xTBase = Float(sigmaT / sigmaS0) * lastSample - Float(alphaT * hPhi1) * m0

        let d1T = modelX0 - m0

        // Gather rks and D1s from history
        var rks: [Double] = []
        var d1s: [MLXArray] = []
        for k in 1..<order {
            let siIdx = i - (k + 1)
            guard siIdx >= 0,
                  let mk = modelOutputs[modelOutputs.count - (k + 1)] else { break }
            let lambdaSk = lambdaOf(Double(sigmas[siIdx]))
            let rk = (lambdaSk - lambdaOf(sigmaS0)) / h
            if rk.isInfinite { break }  // History references sigma=1.0 boundary
            rks.append(rk)
            d1s.append((mk - m0) / Float(rk))
        }
        rks.append(1.0)
        let effectiveOrder = rks.count  // = d1s.count + 1

        // Compute rhos_c coefficients
        let rhosC: [Double]
        if effectiveOrder == 1 {
            rhosC = [0.5]
        } else {
            var hPhiK = hPhi1 / hh - 1.0
            var factorialI = 1.0
            var rRows: [[Double]] = []
            var bVals: [Double] = []
            for j in 1...effectiveOrder {
                rRows.append(rks.map { pow($0, Double(j - 1)) })
                bVals.append(hPhiK * factorialI / bH)
                factorialI *= Double(j + 1)
                hPhiK = hPhiK / hh - 1.0 / factorialI
            }
            rhosC = solveLinearF64(rRows, bVals)
        }

        // Apply correction
        var corrRes = MLXArray.zeros(like: d1T)
        for (kIdx, d1) in d1s.enumerated() {
            corrRes = corrRes + Float(rhosC[kIdx]) * d1
        }
        return xTBase - Float(alphaT * bH) * (corrRes + Float(rhosC.last!) * d1T)
    }

    /// UniPC step: correct current, then predict next.
    public func step(
        modelOutput: MLXArray, timestep: Float, sample: MLXArray
    ) -> MLXArray {
        let i = stepIndex
        var sample = sample

        // Convert velocity -> x0
        let x0 = convertOutput(modelOutput, sample: sample)

        // 1. Corrector: refine current sample if we have history
        let runCorrector = useCorrector && i > 0
            && !disableCorrector.contains(i - 1) && lastSample != nil
        if runCorrector {
            sample = uniCBh2(
                modelX0: x0, lastSample: lastSample!, thisSample: sample,
                order: thisOrder)
        }

        // 2. Shift model output history
        for k in 0..<(solverOrder - 1) {
            modelOutputs[k] = modelOutputs[k + 1]
        }
        modelOutputs[solverOrder - 1] = x0

        // 3. Determine prediction order
        let order = lowerOrderFinal ? min(solverOrder, numSteps - i) : solverOrder
        thisOrder = min(order, lowerOrderNums + 1)

        // 4. Predict next sample
        lastSample = sample
        let xNext = uniPBh2(x0: x0, sample: sample, order: thisOrder)

        if lowerOrderNums < solverOrder {
            lowerOrderNums += 1
        }

        stepIndex += 1
        return xNext
    }

    public func reset() {
        stepIndex = 0
        lowerOrderNums = 0
        modelOutputs = Array(repeating: nil, count: solverOrder)
        lastSample = nil
        thisOrder = 1
    }
}
