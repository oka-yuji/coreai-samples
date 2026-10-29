import Foundation

public enum RAGDocumentFormat: String, Codable, Sendable {
    case html, markdown, text, pdfText, renderedText
}

public struct RAGDocumentSnapshot: Codable, Equatable, Sendable {
    public let id: String
    public let url: String
    public let finalURL: String
    public let title: String
    public let date: String
    public let raw: String
    public let text: String
    public let format: RAGDocumentFormat
    public var structure: RAGDocumentStructure?

    public init(
        id: String, url: String, finalURL: String, title: String, date: String = "", raw: String, text: String,
        format: RAGDocumentFormat
    ) {
        self.id = id
        self.url = url
        self.finalURL = finalURL
        self.title = title
        self.date = date
        self.raw = raw
        self.text = text
        self.format = format
        self.version = RAGContext.fingerprint(text)
    }

    public let version: String

    private enum CodingKeys: String, CodingKey {
        case id, url, finalURL, title, date, raw, text, format, structure, version
    }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: try values.decode(String.self, forKey: .id), url: try values.decode(String.self, forKey: .url),
            finalURL: try values.decode(String.self, forKey: .finalURL),
            title: try values.decode(String.self, forKey: .title),
            date: try values.decode(String.self, forKey: .date), raw: try values.decode(String.self, forKey: .raw),
            text: try values.decode(String.self, forKey: .text),
            format: try values.decode(RAGDocumentFormat.self, forKey: .format))
        structure = try values.decodeIfPresent(RAGDocumentStructure.self, forKey: .structure)
        guard try values.decode(String.self, forKey: .version) == version else {
            throw DecodingError.dataCorruptedError(
                forKey: .version, in: values, debugDescription: "Document version differs from its text.")
        }
    }

    public var sources: [RAGSource] {
        RAGContext.documentSources(title: title, url: url, document: id, date: date, text: text)
    }
}

public struct RAGTextRange: Codable, Equatable, Sendable {
    public var lowerBound: Int
    public var upperBound: Int

    public init(_ lowerBound: Int, _ upperBound: Int) {
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }

    public func text(in original: String) -> String? {
        guard lowerBound >= 0, upperBound >= lowerBound, upperBound <= original.count else { return nil }
        let start = original.index(original.startIndex, offsetBy: lowerBound)
        return String(original[start..<original.index(start, offsetBy: upperBound - lowerBound)])
    }
}

public struct RAGHeading: Equatable, Sendable {
    public var level: Int
    public var title: String
    public var range: RAGTextRange
    public var boundaryStart: Int

    public init(level: Int, title: String, range: RAGTextRange, boundaryStart: Int? = nil) {
        self.level = level
        self.title = title
        self.range = range
        self.boundaryStart = boundaryStart ?? range.lowerBound
    }
}

public struct RAGSection: Codable, Equatable, Sendable {
    public var id: String
    public var level: Int
    public var title: String
    public var heading: RAGTextRange
    public var body: RAGTextRange
    public var parentID: String?
    public var ancestorIDs: [String]
}

public struct RAGDocumentStructure: Codable, Equatable, Sendable {
    public var parserVersion = "document-structure-v1"
    public var sections: [RAGSection]
    public var unavailableReason: String?
    public var content: RAGTextRange?
    public var preamble: RAGTextRange?

    public var isAvailable: Bool { unavailableReason == nil && !sections.isEmpty }

    public static func unavailable(_ reason: String) -> Self {
        Self(sections: [], unavailableReason: reason)
    }

    public static func make(
        documentID: String, text: String, headings: [RAGHeading], content: RAGTextRange? = nil
    ) -> Self {
        let content = content ?? RAGTextRange(0, text.count)
        guard content.text(in: text) != nil else { return .unavailable("invalidContentRange") }
        guard !headings.isEmpty else { return .unavailable("noHeadings") }
        guard headings.count <= 256 else { return .unavailable("headingLimit") }
        let prefix = RAGContext.fingerprint(documentID + "\n" + RAGContext.fingerprint(text))
        var sections: [RAGSection] = []
        var ancestors: [RAGSection] = []
        var previousEnd = content.lowerBound
        for (index, heading) in headings.enumerated() {
            guard (1...6).contains(heading.level),
                !heading.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                heading.range.text(in: text) != nil, heading.range.upperBound > heading.range.lowerBound,
                heading.range.lowerBound >= content.lowerBound,
                heading.boundaryStart >= previousEnd, heading.boundaryStart <= heading.range.lowerBound,
                heading.range.upperBound <= content.upperBound
            else { return .unavailable("invalidHeadingRange") }
            let end = index + 1 < headings.count ? headings[index + 1].boundaryStart : content.upperBound
            guard end >= heading.range.upperBound else { return .unavailable("overlappingHeadings") }
            while let parent = ancestors.last, parent.level >= heading.level { ancestors.removeLast() }
            let section = RAGSection(
                id: prefix + "-" + String(heading.range.lowerBound), level: heading.level, title: heading.title,
                heading: heading.range, body: RAGTextRange(heading.range.upperBound, end),
                parentID: ancestors.last?.id, ancestorIDs: ancestors.map(\.id))
            sections.append(section)
            ancestors.append(section)
            previousEnd = heading.range.upperBound
        }
        let start = headings[0].boundaryStart
        return Self(
            sections: sections, content: content,
            preamble: start > content.lowerBound ? RAGTextRange(content.lowerBound, start) : nil)
    }
}
