import Foundation

public enum ReceiptCategory: String, CaseIterable, Sendable, Codable {
    case travel
    case lodging
    case supplies
    case meals
    case communications
    case books
    case other
    case review

    public var title: String {
        switch self {
        case .travel: "Travel"
        case .lodging: "Lodging"
        case .supplies: "Supplies"
        case .meals: "Meals"
        case .communications: "Communications"
        case .books: "Books and reference materials"
        case .other: "Other"
        case .review: "Needs review"
        }
    }

    public var decisionOption: String {
        switch self {
        case .travel: "Travel: transport fares, taxis, parking and tolls."
        case .lodging: "Lodging: accommodation and hotel stays."
        case .supplies: "Supplies: stationery, office supplies and small equipment."
        case .meals: "Meals: food and drink, including restaurant and cafe purchases."
        case .communications: "Communications: phone, internet, postage and delivery charges."
        case .books: "Books and reference materials: books, journals and learning materials."
        case .other: "Other: a clearly described purchase outside the listed categories."
        case .review: "Needs review: unclear items, mixed categories or insufficient information."
        }
    }
}

public struct ReceiptClassification: Equatable, Sendable {
    public let probabilities: [Float]
    public let topCategory: ReceiptCategory
    public let probability: Float
    public let category: ReceiptCategory

    public init(probabilities: [Float], threshold: Float = 0.75) throws {
        let categories = ReceiptCategory.allCases
        guard probabilities.count == categories.count,
            probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
            abs(probabilities.reduce(0, +) - 1) <= 0.001,
            threshold.isFinite, (0...1).contains(threshold)
        else { throw ReceiptValidationError.invalidClassification }
        var best = 0
        for index in probabilities.indices where probabilities[index] > probabilities[best] { best = index }
        self.probabilities = probabilities
        topCategory = categories[best]
        probability = probabilities[best]
        category = probability >= threshold ? topCategory : .review
    }
}
