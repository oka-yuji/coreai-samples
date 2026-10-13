import Foundation

public enum ProductField: String, CaseIterable, Codable, Sendable {
    case product
    case processor
    case memory
    case storage
    case display
    case weight
    case battery
    case price

    public var title: String {
        switch self {
        case .product: "Product"
        case .processor: "Chip / processor"
        case .memory: "Memory"
        case .storage: "Storage"
        case .display: "Display"
        case .weight: "Weight"
        case .battery: "Battery life"
        case .price: "Price"
        }
    }
}

public enum ProductError: Error, Equatable, LocalizedError {
    case emptyText
    case textTooLong(Int)
    case unsupportedURL
    case invalidEvidence
    case pageTooLarge
    case unsupportedContent
    case http(Int)
    case busy
    case invalidSchema

    public var errorDescription: String? {
        switch self {
        case .emptyText: "No readable product description was found. Paste the description to continue."
        case .textTooLong(let count):
            "This description has \(count) characters. Paste a product section of up to 12,000 characters."
        case .unsupportedURL: "Use an HTTPS product page from Apple or Microsoft, or paste a description."
        case .invalidEvidence: "An extracted value did not match its source text. Please try again."
        case .pageTooLarge: "This page is too large. Paste the product description instead."
        case .unsupportedContent: "This URL does not contain an HTML or text page. Paste the description instead."
        case .http(let code): "The website returned HTTP \(code). Open it in your browser and paste the description."
        case .busy: "The previous extraction is finishing. Please try again shortly."
        case .invalidSchema: "The product extraction configuration could not be loaded."
        }
    }
}

public struct ProductSource: Equatable, Sendable {
    public static let textLimit = 12_000
    public let title: String
    public let text: String
    public let url: URL?
    public let capturedAt: Date

    public init(title: String, text: String, url: URL?, capturedAt: Date) throws {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw ProductError.emptyText }
        guard text.unicodeScalars.count <= Self.textLimit else {
            throw ProductError.textTooLong(text.unicodeScalars.count)
        }
        self.title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        self.text = text
        self.url = url
        self.capturedAt = capturedAt
    }
}

public struct ProductCandidate: Equatable, Sendable {
    public let field: ProductField
    public let range: Range<Int>
    public let text: String
    public let probability: Float

    public init(field: ProductField, range: Range<Int>, text: String, probability: Float) {
        self.field = field
        self.range = range
        self.text = text
        self.probability = probability
    }
}

public struct ProductEvidence: Equatable, Sendable {
    public let candidate: ProductCandidate
    public let context: String
    public let contextRange: Range<Int>

    public init(candidate: ProductCandidate, context: String, contextRange: Range<Int>) {
        self.candidate = candidate
        self.context = context
        self.contextRange = contextRange
    }
}

public struct ProductMemo: Equatable, Sendable {
    public let source: ProductSource
    public let evidence: [ProductEvidence]

    public init(source: ProductSource, candidates: [ProductCandidate]) throws {
        let scalars = Array(source.text.unicodeScalars)
        var unique: [ProductCandidate] = []
        for candidate in candidates {
            guard candidate.range.lowerBound >= 0, candidate.range.upperBound <= scalars.count,
                !candidate.range.isEmpty, candidate.probability.isFinite,
                (0...1).contains(candidate.probability),
                Array(String(String.UnicodeScalarView(scalars[candidate.range])).utf8) == Array(candidate.text.utf8)
            else { throw ProductError.invalidEvidence }
            if let index = unique.firstIndex(where: { $0.field == candidate.field && $0.range == candidate.range }) {
                if unique[index].probability < candidate.probability { unique[index] = candidate }
            } else {
                unique.append(candidate)
            }
        }
        self.source = source
        evidence = unique.sorted {
            ($0.range.lowerBound, $0.range.upperBound, $0.field.rawValue)
                < ($1.range.lowerBound, $1.range.upperBound, $1.field.rawValue)
        }.map { candidate in
            var lower = candidate.range.lowerBound
            var upper = candidate.range.upperBound
            while lower > 0 {
                if lower > 1 && scalars[lower - 1] == "\n" && scalars[lower - 2] == "\n" { break }
                lower -= 1
            }
            while upper < scalars.count {
                if upper + 1 < scalars.count && scalars[upper] == "\n" && scalars[upper + 1] == "\n" { break }
                upper += 1
            }
            let range = lower..<upper
            return ProductEvidence(
                candidate: candidate, context: String(String.UnicodeScalarView(scalars[range])), contextRange: range)
        }
    }

    public func values(for field: ProductField) -> [ProductEvidence] {
        evidence.filter { $0.candidate.field == field }
    }
}
