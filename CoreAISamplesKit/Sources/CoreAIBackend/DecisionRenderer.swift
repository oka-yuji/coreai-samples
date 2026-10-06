import Foundation
import Tokenizers

public struct DecisionLimits: Sendable, Hashable {
    public var questions: Int
    public var optionsPerQuestion: Int
    public var tokens: Int

    public init(questions: Int = 16, optionsPerQuestion: Int = 8, tokens: Int = 9_216) {
        self.questions = questions
        self.optionsPerQuestion = optionsPerQuestion
        self.tokens = tokens
    }
}

public enum DecisionError: Error, Equatable, LocalizedError {
    case noQuestions
    case duplicateQuestion(String)
    case noOptions(question: String)
    case tooManyQuestions(count: Int, limit: Int)
    case tooManyOptions(question: String, count: Int, limit: Int)
    case tooManyTokens(count: Int, limit: Int)

    public var errorDescription: String? {
        switch self {
        case .noQuestions: "The request has no questions."
        case .duplicateQuestion(let id): "The question \(id) appears more than once."
        case .noOptions(let id): "The question \(id) has no options."
        case .tooManyQuestions(let count, let limit):
            "The request has \(count) questions; the model takes at most \(limit) questions."
        case .tooManyOptions(let id, let count, let limit):
            "The question \(id) has \(count) options; the model takes at most \(limit) options per question."
        case .tooManyTokens(let count, let limit):
            "The request renders to \(count) tokens; the model takes at most \(limit) tokens."
        }
    }
}

public struct RenderedDecision: Sendable, Hashable {
    public let tokenIDs: [Int]
    public let questionSpans: [Range<Int>]
    public let optionSpans: [[Range<Int>]]
    public let optionIDs: [[String]]
    public let questionTypes: [Int]
}

public enum DecisionRenderer {
    struct Fragment {
        enum Role {
            case question(Int)
            case option(Int)
        }

        var text: String
        var role: Role? = nil
    }

    static let systemPrompt =
        "Read the complete state and schema. Decide every field jointly. "
        + "Each answer must be exactly one of that field's allowed options."

    public static func render(
        _ request: DecisionRequest,
        tokenizer: any Tokenizer,
        limits: DecisionLimits = DecisionLimits()
    ) throws -> RenderedDecision {
        try render(request, limits: limits) { tokenizer.encode(text: $0, addSpecialTokens: false) }
    }

    static func render(
        _ request: DecisionRequest,
        limits: DecisionLimits,
        encode: (String) -> [Int]
    ) throws -> RenderedDecision {
        try validate(request, limits: limits)
        var ids: [Int] = []
        var questionSpans: [Range<Int>] = []
        var optionSpans = [[Range<Int>]](repeating: [], count: request.questions.count)
        for fragment in fragments(for: request) {
            let start = ids.count
            ids += encode(fragment.text)
            switch fragment.role {
            case .question: questionSpans.append(start..<ids.count)
            case .option(let question): optionSpans[question].append(start..<ids.count)
            case nil: break
            }
        }
        guard ids.count <= limits.tokens else {
            throw DecisionError.tooManyTokens(count: ids.count, limit: limits.tokens)
        }
        return RenderedDecision(
            tokenIDs: ids,
            questionSpans: questionSpans,
            optionSpans: optionSpans,
            optionIDs: request.questions.map { options(of: $0).map(\.id) },
            questionTypes: request.questions.map(\.criteria.code)
        )
    }

    static func validate(_ request: DecisionRequest, limits: DecisionLimits) throws {
        let questions = request.questions
        guard !questions.isEmpty else { throw DecisionError.noQuestions }
        guard questions.count <= limits.questions else {
            throw DecisionError.tooManyQuestions(count: questions.count, limit: limits.questions)
        }
        var seen: Set<String> = []
        for question in questions {
            guard seen.insert(question.id).inserted else { throw DecisionError.duplicateQuestion(question.id) }
            let count = options(of: question).count
            guard count > 0 else { throw DecisionError.noOptions(question: question.id) }
            guard count <= limits.optionsPerQuestion else {
                throw DecisionError.tooManyOptions(
                    question: question.id, count: count, limit: limits.optionsPerQuestion)
            }
        }
    }

    static func fragments(for request: DecisionRequest) -> [Fragment] {
        var fragments = [
            Fragment(text: "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\nSTATE:\n"),
            Fragment(text: request.state),
            Fragment(text: "\n\nSCHEMA FIELDS:\n"),
        ]
        for (index, question) in request.questions.enumerated() {
            let instructions = question.instructions ?? ""
            fragments += [
                Fragment(
                    text: "\nFIELD \(index + 1)\nID: \(question.id)\nTYPE: \(question.criteria.name)\nINSTRUCTION: "),
                Fragment(text: instructions.isEmpty ? question.id : instructions, role: .question(index)),
                Fragment(text: "\nALLOWED OPTIONS:\n"),
            ]
            for (number, option) in options(of: question).enumerated() {
                fragments += [
                    Fragment(text: "OPTION \(number + 1): "),
                    Fragment(text: semantics(option), role: .option(index)),
                    Fragment(text: "\n"),
                ]
            }
            fragments.append(Fragment(text: "END FIELD\n"))
        }
        fragments.append(
            Fragment(text: "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"))
        return fragments
    }

    static func options(of question: DecisionQuestion) -> [(id: String, description: String?)] {
        switch question.criteria {
        case .noul(let criteria):
            [
                (id: "true", description: criteria["true"] ?? "The proposition is true or the answer is yes."),
                (id: "false", description: criteria["false"] ?? "The proposition is false or the answer is no."),
            ]
        case .choice(let criteria):
            criteria.map { (id: $0.key, description: $0.value) }
                .sorted { $0.id.unicodeScalars.lexicographicallyPrecedes($1.id.unicodeScalars) }
        case .score(let levels):
            levels.enumerated().map { (id: String($0.offset), description: $0.element) }
        }
    }

    static func semantics(_ option: (id: String, description: String?)) -> String {
        let id = "\"option_id\":" + jsonString(option.id)
        guard let description = option.description else { return "{\(id)}" }
        return "{\"description\":\(jsonString(description)),\(id)}"
    }

    static func jsonString(_ text: String) -> String {
        var escaped = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": escaped += "\\\""
            case "\\": escaped += "\\\\"
            case "\n": escaped += "\\n"
            case "\r": escaped += "\\r"
            case "\t": escaped += "\\t"
            case "\u{08}": escaped += "\\b"
            case "\u{0C}": escaped += "\\f"
            case "\u{00}"..."\u{1F}":
                escaped += (scalar.value < 0x10 ? "\\u000" : "\\u00") + String(scalar.value, radix: 16)
            default: escaped.unicodeScalars.append(scalar)
            }
        }
        return escaped + "\""
    }
}

extension DecisionQuestion.Criteria {
    var name: String {
        switch self {
        case .noul: "noul"
        case .choice: "choice"
        case .score: "score"
        }
    }

    var code: Int {
        switch self {
        case .noul: 0
        case .choice: 1
        case .score: 2
        }
    }
}
