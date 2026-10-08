public struct FormSchema: Codable, Equatable, Sendable {
    public var fields: [FormField]

    public init(fields: [FormField]) {
        self.fields = fields
    }

    public func extractionSchema() -> GLiNERSchema {
        .entities(fields.map { EntityLabel(name: $0.modelLabel, description: $0.description) })
    }

    public func classificationSchema(for field: FormField) -> GLiNERSchema {
        guard case .list(let options, let modelTask) = field.kind else {
            return .classification(name: field.name, labels: [])
        }
        return .classification(name: modelTask ?? field.name, labels: options.map(\.modelLabel))
    }
}

public struct FormField: Codable, Equatable, Sendable {
    public enum Kind: Codable, Equatable, Sendable {
        case number(unit: String)

        case list(options: [FormOption], modelTask: String? = nil)
    }

    public var key: String

    public var name: String

    public var modelLabel: String
    public var kind: Kind

    public var description: String?

    public init(key: String, name: String, modelLabel: String? = nil, kind: Kind, description: String? = nil) {
        self.key = key
        self.name = name
        self.modelLabel = modelLabel ?? name
        self.kind = kind
        self.description = description
    }
}

public struct FormOption: Codable, Equatable, Sendable {

    public var name: String

    public var modelLabel: String

    public init(name: String, modelLabel: String? = nil) {
        self.name = name
        self.modelLabel = modelLabel ?? name
    }
}
