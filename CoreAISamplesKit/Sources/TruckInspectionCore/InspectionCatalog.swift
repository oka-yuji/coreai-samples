import Foundation

public enum TruckProfile: String, Codable, CaseIterable, Sendable {
    case hydraulic, airBrake
}

public enum InspectionStatus: String, Codable, CaseIterable, Sendable {
    case ok, attention, unchecked, notApplicable
}

public struct InspectionChoice: Codable, Equatable, Sendable {
    public let status: InspectionStatus
    public let title: String
    public let modelLabel: String
    public let aliases: [String]
}

public struct InspectionItem: Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let title: String
    public let group: String
    public let question: String
    public let guidance: String
    public let modelTask: String
    public let profiles: [TruckProfile]
    public let conditionBased: Bool
    public let aliases: [String: [String]]
}

public struct InspectionVocabulary: Codable, Equatable, Sendable {
    public let yes: [String]
    public let no: [String]
    public let retry: String
    public let confirm: String
    public let acknowledged: String
    public let completed: String
}

public struct InspectionSource: Codable, Equatable, Sendable {
    public let title: String
    public let url: URL
    public let checkedOn: String
}

public struct InspectionCatalog: Codable, Equatable, Sendable {
    public let version: Int
    public let title: String
    public let notice: String
    public let profiles: [String: String]
    public let choices: [InspectionChoice]
    public let vocabulary: InspectionVocabulary
    public let items: [InspectionItem]
    public let sources: [InspectionSource]

    public static func decode(_ data: Data) throws -> Self {
        let catalog = try JSONDecoder().decode(Self.self, from: data)
        try catalog.validate()
        return catalog
    }

    public func items(for profile: TruckProfile) -> [InspectionItem] {
        items.filter { $0.profiles.contains(profile) }
    }

    public func title(for status: InspectionStatus) -> String {
        choices.first { $0.status == status }?.title ?? status.rawValue
    }

    public func prompt(for item: InspectionItem, retry: Bool = false) -> String {
        (retry ? vocabulary.retry : "") + item.question
    }

    public func confirmation(for item: InspectionItem, status: InspectionStatus) -> String {
        vocabulary.confirm.replacingOccurrences(of: "{item}", with: item.title)
            .replacingOccurrences(of: "{answer}", with: title(for: status))
    }

    public func acknowledgment(for item: InspectionItem, status: InspectionStatus) -> String {
        vocabulary.acknowledged.replacingOccurrences(of: "{item}", with: item.title)
            .replacingOccurrences(of: "{answer}", with: title(for: status))
    }

    public func hints(for item: InspectionItem?, confirming: Bool) -> [String] {
        if confirming { return vocabulary.yes + vocabulary.no }
        return choices.flatMap { [$0.title] + $0.aliases } + (item?.aliases.values.flatMap { $0 } ?? [])
    }

    private func validate() throws {
        guard version == 1, !items.isEmpty, Set(items.map(\.id)).count == items.count,
            choices.count == InspectionStatus.allCases.count,
            Set(choices.map(\.status)) == Set(InspectionStatus.allCases),
            Set(choices.map(\.modelLabel)).count == choices.count,
            TruckProfile.allCases.allSatisfy({ profiles[$0.rawValue] != nil && !items(for: $0).isEmpty }),
            !vocabulary.yes.isEmpty, !vocabulary.no.isEmpty, !sources.isEmpty
        else { throw InspectionError.invalidCatalog }
        for item in items {
            guard !item.id.isEmpty, !item.title.isEmpty, !item.question.isEmpty, !item.modelTask.isEmpty,
                !item.profiles.isEmpty, item.aliases.keys.allSatisfy({ InspectionStatus(rawValue: $0) != nil })
            else { throw InspectionError.invalidCatalog }
            var ownership: [String: InspectionStatus] = [:]
            for choice in choices {
                let words = [choice.title] + choice.aliases + (item.aliases[choice.status.rawValue] ?? [])
                for word in words {
                    let normalized = InspectionRules.normalize(word)
                    guard !normalized.isEmpty,
                        ownership[normalized] == nil || ownership[normalized] == choice.status
                    else { throw InspectionError.invalidCatalog }
                    ownership[normalized] = choice.status
                }
            }
        }
    }
}

public enum InspectionError: Error, Equatable {
    case invalidCatalog, invalidClassification, emptyAnswer, answerTooLong, noModel, busy, unavailable
}
