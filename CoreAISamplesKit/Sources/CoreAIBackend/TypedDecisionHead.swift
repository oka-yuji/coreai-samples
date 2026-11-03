import Foundation

public struct TypedDecisionHead: Sendable {
    private struct Tensor: Decodable {
        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case dataOffsets = "data_offsets"
        }

        let dtype: String
        let shape: [Int]
        let dataOffsets: [Int]
    }

    private struct Header: Decodable {
        enum CodingKeys: String, CodingKey {
            case weight = "proj.weight"
            case bias = "proj.bias"
        }

        let weight: Tensor
        let bias: Tensor
    }

    static let rows = 24
    static let slots: [TypedDecisionKind: Range<Int>] = [.noul: 0..<2, .score: 2..<8, .choice: 8..<24]

    public let width: Int
    private let data: Data
    private let weightOffset: Int
    private let biasOffset: Int

    public init(url: URL, width: Int) throws {
        do {
            let data = try Data(contentsOf: url, options: .alwaysMapped)
            guard data.count >= 8 else { throw TypedDecisionError.invalidBundle("The head header is truncated.") }
            let size = data.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
            guard size <= UInt64(data.count - 8), size <= 16 * 1_024 * 1_024 else {
                throw TypedDecisionError.invalidBundle("The head header size is invalid.")
            }
            let start = 8 + Int(size)
            let available = data.count - start
            guard width > 0, width <= available / (Self.rows * 4) else {
                throw TypedDecisionError.invalidBundle("The head file is too small for width \(width).")
            }
            let header = try JSONDecoder().decode(Header.self, from: data.subdata(in: 8..<start))
            weightOffset = try Self.offset(
                of: header.weight, shape: [Self.rows, width], start: start, available: available)
            biasOffset = try Self.offset(of: header.bias, shape: [Self.rows], start: start, available: available)
            self.data = data
            self.width = width
        } catch let error as TypedDecisionError {
            throw error
        } catch {
            throw TypedDecisionError.invalidBundle(error.localizedDescription)
        }
    }

    public func logits(hidden: [Float]) throws -> [Float] {
        guard hidden.count == width else {
            throw TypedDecisionError.invalidHidden(count: hidden.count, expected: width)
        }
        return data.withUnsafeBytes { raw in
            (0..<Self.rows).map { row in
                let base = weightOffset + row * width * 4
                var sum: Float = 0
                for column in 0..<width {
                    sum += Self.float(raw, at: base + column * 4) * hidden[column]
                }
                return sum + Self.float(raw, at: biasOffset + row * 4)
            }
        }
    }

    public func probabilities(logits: [Float], kind: TypedDecisionKind, count: Int, temperature: Float) throws -> (
        shipping: [Float], plain: [Float]
    ) {
        guard logits.count == Self.rows else {
            throw TypedDecisionError.invalidHidden(count: logits.count, expected: Self.rows)
        }
        guard logits.allSatisfy(\.isFinite) else {
            throw TypedDecisionError.nonFinite("The head logits must be finite.")
        }
        guard temperature.isFinite, temperature > 0 else {
            throw TypedDecisionError.nonFinite("The temperature must be a positive finite number.")
        }
        guard let slot = Self.slots[kind], kind == .choice ? (2...slot.count).contains(count) : count == slot.count
        else { throw TypedDecisionError.invalidOptions("\(kind.rawValue) cannot take \(count) options.") }
        let selected = logits[slot.lowerBound..<(slot.lowerBound + count)].map(Double.init)
        let scaled = selected.map { $0 / Double(temperature) }
        guard scaled.allSatisfy(\.isFinite) else {
            throw TypedDecisionError.nonFinite("The head logits divided by the temperature must be finite.")
        }
        return (Self.softmax(scaled), Self.softmax(selected))
    }

    public static func top(of values: [Float]) -> Int {
        var best = 0
        for index in values.indices where values[index] > values[best] {
            best = index
        }
        return best
    }

    private static func softmax(_ values: [Double]) -> [Float] {
        let maximum = values.max() ?? 0
        let weights = values.map { exp($0 - maximum) }
        let total = weights.reduce(0, +)
        return weights.map { Float($0 / total) }
    }

    private static func float(_ raw: UnsafeRawBufferPointer, at offset: Int) -> Float {
        Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
    }

    private static func offset(of tensor: Tensor, shape: [Int], start: Int, available: Int) throws -> Int {
        guard tensor.dtype == "F32", tensor.shape == shape, tensor.dataOffsets.count == 2,
            tensor.dataOffsets[0] >= 0, tensor.dataOffsets[1] <= available,
            tensor.dataOffsets[1] >= tensor.dataOffsets[0],
            tensor.dataOffsets[1] - tensor.dataOffsets[0] == shape.reduce(4, *),
            (start + tensor.dataOffsets[0]).isMultiple(of: 4)
        else {
            throw TypedDecisionError.invalidBundle(
                "The head must hold float32 proj.weight [24, width] and proj.bias [24] inside the file.")
        }
        return start + tensor.dataOffsets[0]
    }
}
