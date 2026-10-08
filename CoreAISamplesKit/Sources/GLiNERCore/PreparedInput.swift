public struct WordSpan: Equatable, Sendable {
    public var text: String
    public var scalarRange: Range<Int>

    public init(text: String, scalarRange: Range<Int>) {
        self.text = text
        self.scalarRange = scalarRange
    }
}

public struct EntityLabel: Hashable, Sendable {
    public var name: String
    public var description: String?

    public init(name: String, description: String? = nil) {
        self.name = name
        self.description = description
    }
}

public enum GLiNERSchema: Hashable, Sendable {
    case entities([EntityLabel])
    case classification(name: String, labels: [String])
}

public enum GLiNERPreprocessingError: Error, Equatable {
    case inputTooLong(tokens: Int, limit: Int)
    case tooManyLabels(count: Int, limit: Int)
}

public struct PreparedInput: Equatable, Sendable {
    public static let tokenCapacities = [256, 512]
    public static let labelCapacity = 16

    public var inputIDs: [Int32]
    public var attentionMask: [Int32]
    public var wordIndex: [Int32]
    public var wordMask: [Int32]
    public var queryIndex: [Int32]
    public var queryMask: [Int32]
    public var classIndex: [Int32]
    public var classMask: [Int32]
    public var tokenCapacity: Int

    public var words: [WordSpan]
    public var labelCount: Int
    public var appendedPeriod: Bool

    public init(
        text: String,
        schema: GLiNERSchema,
        tokenCapacities: [Int] = PreparedInput.tokenCapacities,
        splitWords: (String) -> [WordSpan],
        tokenize: (String) -> [Int32]
    ) throws {
        let endsWithStop = text.unicodeScalars.last.map { ".!?".unicodeScalars.contains($0) } ?? false
        let words = splitWords(endsWithStop ? text : text + ".")

        let prompt: String
        let marker: String
        let labels: [String]
        let isClassification: Bool
        switch schema {
        case .entities(let entities):
            prompt = entities.reduce("entities") { prompt, entity in
                entity.description.map { "\(prompt) [DESCRIPTION] \(entity.name): \($0)" } ?? prompt
            }
            marker = "[E]"
            labels = entities.map(\.name)
            isClassification = false
        case .classification(let name, let classLabels):
            prompt = name
            marker = "[L]"
            labels = classLabels
            isClassification = true
        }
        guard labels.count <= Self.labelCapacity else {
            throw GLiNERPreprocessingError.tooManyLabels(count: labels.count, limit: Self.labelCapacity)
        }

        var ids = ["(", "[P]", prompt, "("].flatMap(tokenize)
        var markerPositions: [Int32] = []
        for label in labels {
            markerPositions.append(Int32(ids.count))
            ids += tokenize(marker) + tokenize(label)
        }
        ids += [")", ")", "[SEP_TEXT]"].flatMap(tokenize)
        var wordPositions: [Int32] = []
        for word in words {
            wordPositions.append(Int32(ids.count))
            ids += tokenize(word.text)
        }

        let needed = max(ids.count, words.count)
        guard let capacity = tokenCapacities.filter({ needed <= $0 }).min() else {
            throw GLiNERPreprocessingError.inputTooLong(tokens: ids.count, limit: tokenCapacities.max() ?? 0)
        }
        let markerIndex = Self.padded(markerPositions, to: Self.labelCapacity)
        let markerMask = Self.mask(count: labels.count, capacity: Self.labelCapacity)
        let none = Self.padded([], to: Self.labelCapacity)

        inputIDs = Self.padded(ids, to: capacity)
        attentionMask = Self.mask(count: ids.count, capacity: capacity)
        wordIndex = Self.padded(wordPositions, to: capacity)
        wordMask = Self.mask(count: words.count, capacity: capacity)
        queryIndex = isClassification ? none : markerIndex
        queryMask = isClassification ? none : markerMask
        classIndex = isClassification ? markerIndex : none
        classMask = isClassification ? markerMask : none
        tokenCapacity = capacity
        self.words = words
        labelCount = labels.count
        appendedPeriod = !endsWithStop
    }

    private static func padded(_ values: [Int32], to capacity: Int) -> [Int32] {
        values + Array(repeating: 0, count: capacity - values.count)
    }

    private static func mask(count: Int, capacity: Int) -> [Int32] {
        Array(repeating: 1, count: count) + Array(repeating: 0, count: capacity - count)
    }
}
