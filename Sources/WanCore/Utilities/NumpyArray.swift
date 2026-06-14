//
//  NumpyArray.swift
//
//  Minimal .npy file reader for the parity tests. Supports the format the
//  Python fixture-dump script produces: little-endian fp32, version 1.0,
//  C-order. No support for fortran_order, structured dtypes, or other
//  variants — keep it boring.
//
//  Format reference: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
//

import Foundation
import MLX

public enum NumpyError: LocalizedError {
    case badMagic
    case unsupportedDescr(String)
    case fortranOrderUnsupported
    case headerParseFailure(String)

    public var errorDescription: String? {
        switch self {
        case .badMagic: return "Not a .npy file (magic bytes mismatch)"
        case .unsupportedDescr(let s): return "Unsupported numpy dtype: \(s) (only '<f4' is supported)"
        case .fortranOrderUnsupported: return "Fortran-order .npy files are not supported"
        case .headerParseFailure(let s): return "Could not parse .npy header: \(s)"
        }
    }
}

/// Load a `.npy` file produced by `numpy.save` and return it as an MLXArray.
/// Restricted to little-endian fp32 / C-order — what our fixture-dump
/// script writes.
public func loadNumpy(url: URL) throws -> MLXArray {
    let data = try Data(contentsOf: url)
    // Magic: \x93NUMPY
    guard data.count > 10,
          data[0] == 0x93,
          data[1...5] == Data("NUMPY".utf8) else {
        throw NumpyError.badMagic
    }

    let major = data[6]
    let _ = data[7]   // minor — we don't gate on it

    var headerLen: Int
    var headerStart: Int
    if major == 1 {
        // 2-byte little-endian header length follows.
        headerLen = Int(data[8]) | (Int(data[9]) << 8)
        headerStart = 10
    } else {
        // 4-byte little-endian header length for v2/v3.
        headerLen = (0..<4).reduce(0) { acc, i in acc | (Int(data[8 + i]) << (8 * i)) }
        headerStart = 12
    }

    let headerData = data[headerStart..<(headerStart + headerLen)]
    guard let header = String(data: headerData, encoding: .utf8) else {
        throw NumpyError.headerParseFailure("non-utf8 header")
    }

    // Header is a Python dict literal — parse the three keys we need.
    func extract(_ key: String) -> String? {
        guard let r = header.range(of: "'\(key)':") else { return nil }
        let tail = header[r.upperBound...]
        if key == "shape" {
            // 'shape': ( ... ) — extract the parenthesized tuple verbatim
            // so internal commas don't confuse the regular path.
            guard let openParen = tail.firstIndex(of: "("),
                  let closeParen = tail.firstIndex(of: ")") else { return nil }
            return String(tail[openParen...closeParen])
        }
        // Other values: extract up to the next ',' or '}'.
        let stop = tail.firstIndex(where: { $0 == "," || $0 == "}" }) ?? tail.endIndex
        return String(tail[..<stop]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    guard let descr = extract("descr")?.trimmingCharacters(in: CharacterSet(charactersIn: "' "))
    else {
        throw NumpyError.headerParseFailure("missing 'descr'")
    }
    // Supported dtypes: <f4 (fp32) and <i4 (int32). Other dtypes need an
    // explicit case here — generic numpy support is out of scope.
    let supported = ["<f4", "<i4"]
    guard supported.contains(descr) else {
        throw NumpyError.unsupportedDescr(descr)
    }

    let fortranToken = extract("fortran_order") ?? "False"
    if fortranToken.lowercased().contains("true") {
        throw NumpyError.fortranOrderUnsupported
    }

    guard let shapeRaw = extract("shape") else {
        throw NumpyError.headerParseFailure("missing 'shape'")
    }
    // shapeRaw looks like "(1, 16, 2, 2, 2,)" — parse the comma-separated ints.
    let nums = shapeRaw
        .trimmingCharacters(in: CharacterSet(charactersIn: "() "))
        .split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    guard !nums.isEmpty else {
        throw NumpyError.headerParseFailure("empty shape: \(shapeRaw)")
    }

    // Payload: row-major little-endian, fp32 or int32 (4 bytes either way).
    let payload = data[(headerStart + headerLen)...]
    let count = nums.reduce(1, *)
    let expectedBytes = count * 4   // both fp32 and int32 are 4 bytes/element
    guard payload.count == expectedBytes else {
        throw NumpyError.headerParseFailure(
            "payload size mismatch: have \(payload.count), expected \(expectedBytes)"
        )
    }

    if descr == "<f4" {
        let floats: [Float] = payload.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        return MLXArray(floats, nums)
    } else {
        // <i4 — int32
        let ints: [Int32] = payload.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Int32.self))
        }
        return MLXArray(ints, nums)
    }
}
