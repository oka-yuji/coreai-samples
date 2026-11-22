import ArgumentParser
import Foundation

struct EvaluationInput: Decodable, Equatable {
    let id: String
    let inputIds: [Int32]
    let inputSha256: String
    let maxNewTokens: Int

    static func read(_ data: Data, vocabulary: Int) throws -> [EvaluationInput] {
        var items: [EvaluationInput] = []
        var seen: Set<String> = []
        for (index, line) in data.split(separator: UInt8(ascii: "\n")).enumerated() {
            let item: EvaluationInput
            do {
                item = try JSONDecoder.snake.decode(EvaluationInput.self, from: line)
            } catch {
                throw ValidationError("Input record \(index + 1) could not be read: \(error)")
            }
            guard seen.insert(item.id).inserted else {
                throw ValidationError("The input id \(item.id) appears more than once.")
            }
            guard !item.inputIds.isEmpty, item.inputIds.allSatisfy({ (0..<Int32(vocabulary)).contains($0) }) else {
                throw ValidationError(
                    "\(item.id): input_ids must hold at least one token id from 0 to \(vocabulary - 1).")
            }
            guard item.maxNewTokens > 0 else { throw ValidationError("\(item.id): max_new_tokens must be positive.") }
            guard tokensDigest(item.inputIds.map(Int.init)) == item.inputSha256 else {
                throw ValidationError("\(item.id): input_sha256 differs from the SHA-256 of input_ids.")
            }
            items.append(item)
        }
        guard !items.isEmpty else { throw ValidationError("The input file has no records.") }
        return items
    }

    func generate(with source: some LogitsSource, stopTokens: Set<Int32>) async throws -> OutputRecord {
        var record = OutputRecord(id: id, inputSha256: inputSha256, inputTokens: inputIds.count)
        source.reset()
        do {
            try await source.prefill(inputIds, usePrefillFunction: true)
            while true {
                switch LogitsRow(source.logits()) {
                case .nonFinite(let count):
                    record.nonFinite = count
                    record.error = "The logits had \(count) non-finite values at output position \(record.top2.count)."
                    return record
                case .finite(let top):
                    record.top2.append(top)
                    if stopTokens.contains(top.first) {
                        record.stop = "stop_token"
                        record.stopToken = top.first
                        return record
                    }
                    record.outputIds.append(top.first)
                    if record.outputIds.count >= maxNewTokens { return record }
                    try await source.step(top.first)
                }
            }
        } catch let error as CancellationError {
            throw error
        } catch {
            record.error = "\(type(of: error)): \(error)"
            return record
        }
    }
}
