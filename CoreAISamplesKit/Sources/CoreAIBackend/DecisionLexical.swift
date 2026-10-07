import Foundation

struct DecisionLexical: Sendable {
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
        enum CodingKeys: String, CodingKey { case weight = "lm_head.weight" }
        let weight: Tensor
    }

    let data: Data
    let base: Int
    let rows: Int
    let width: Int

    init(url: URL, width: Int) throws {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 8 else { throw DecisionError.invalidBundle("The lexical header is truncated.") }
        let size = data.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        guard size <= UInt64(data.count - 8), size <= 16 * 1_024 * 1_024 else {
            throw DecisionError.invalidBundle("The lexical header size is invalid.")
        }
        let headerEnd = 8 + Int(size)
        let tensor = try JSONDecoder().decode(Header.self, from: data.subdata(in: 8..<headerEnd)).weight
        guard tensor.dtype == "BF16",
            tensor.shape.count == 2, tensor.shape[0] > 0, tensor.shape[1] == width,
            tensor.shape[0] <= (data.count - headerEnd) / (width * 2),
            tensor.dataOffsets.count == 2, tensor.dataOffsets[0] >= 0,
            tensor.dataOffsets[1] >= tensor.dataOffsets[0],
            tensor.dataOffsets[1] <= data.count - headerEnd,
            tensor.dataOffsets[1] - tensor.dataOffsets[0] == tensor.shape[0] * width * 2,
            (headerEnd + tensor.dataOffsets[0]).isMultiple(of: 2)
        else { throw DecisionError.invalidBundle("The lexical tensor must contain BF16 lm_head.weight rows.") }
        self.data = data
        self.base = headerEnd + tensor.dataOffsets[0]
        self.rows = tensor.shape[0]
        self.width = width
    }

    func vectors(rendered: RenderedDecision, questions: Int, options: Int) throws -> [Float16] {
        var output = [Float16](repeating: 0, count: questions * options * width)
        var accumulated = [Float](repeating: 0, count: width)
        try data.withUnsafeBytes { raw in
            let values = raw.bindMemory(to: UInt16.self)
            for (question, spans) in rendered.optionSpans.enumerated() {
                try Task.checkCancellation()
                for (option, span) in spans.enumerated() {
                    guard !span.isEmpty else { throw DecisionError.invalidRequest("An option has no tokens.") }
                    accumulated.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0) }
                    for token in span {
                        let id = rendered.tokenIDs[token]
                        guard id >= 0, id < rows else {
                            throw DecisionError.invalidRequest("A token is outside the lexical vocabulary.")
                        }
                        let start = base / 2 + id * width
                        for column in 0..<width {
                            accumulated[column] += Float(
                                bitPattern: UInt32(UInt16(littleEndian: values[start + column])) << 16)
                        }
                    }
                    let start = (question * options + option) * width
                    for column in 0..<width {
                        output[start + column] = Float16(accumulated[column] / Float(span.count))
                    }
                }
            }
        }
        return output
    }
}
