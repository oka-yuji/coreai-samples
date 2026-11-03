import Foundation
import Tokenizers

public enum TypedDecisionRenderer {
    static let template = "[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n{options}\n[decision]:"
    static let labels = "ABCDEFGHIJKLMNOP"
    static let fixedOptions: [TypedDecisionKind: [String]] = [
        .noul: ["false", "true"], .score: ["0", "1", "2", "3", "4", "5"],
    ]
    static let lineBreaks: Set<Unicode.Scalar> = [
        "\u{0A}", "\u{0B}", "\u{0C}", "\u{0D}", "\u{1C}", "\u{1D}", "\u{1E}", "\u{85}", "\u{2028}", "\u{2029}",
    ]

    public static func render(_ request: TypedDecisionRequest) throws -> String {
        let lines = try optionLines(kind: request.kind, options: request.options)
        return "[kind] \(request.kind.rawValue)\n[state] \(request.state)\n[question] \(request.question)\n[options]\n"
            + lines.joined(separator: "\n") + "\n[decision]:"
    }

    public static func tokenIDs(
        for request: TypedDecisionRequest, tokenizer: any Tokenizer, limit: Int
    ) throws -> [Int] {
        try tokenIDs(for: request, limit: limit) { tokenizer.encode(text: $0, addSpecialTokens: false) }
    }

    static func tokenIDs(for request: TypedDecisionRequest, limit: Int, encode: (String) -> [Int]) throws -> [Int] {
        let ids = encode(try render(request))
        guard ids.count <= limit else { throw TypedDecisionError.tooManyTokens(count: ids.count, limit: limit) }
        return ids
    }

    static func optionLines(kind: TypedDecisionKind, options: [String]) throws -> [String] {
        if let fixed = fixedOptions[kind] {
            guard options.map({ Array($0.utf8) }) == fixed.map({ Array($0.utf8) }) else {
                throw TypedDecisionError.invalidOptions(
                    "\(kind.rawValue) accepts only \(fixed.joined(separator: ", ")) in this order.")
            }
            return options
        }
        guard (2...labels.count).contains(options.count) else {
            throw TypedDecisionError.invalidOptions("choice needs 2 to \(labels.count) options, not \(options.count).")
        }
        guard options.allSatisfy({ !$0.unicodeScalars.contains(where: lineBreaks.contains) }) else {
            throw TypedDecisionError.invalidOptions("choice options must not contain a line break.")
        }
        guard Set(options.map { Array($0.utf8) }).count == options.count else {
            throw TypedDecisionError.invalidOptions("choice options must be distinct.")
        }
        return zip(labels, options).map { "\($0)) \($1)" }
    }
}
