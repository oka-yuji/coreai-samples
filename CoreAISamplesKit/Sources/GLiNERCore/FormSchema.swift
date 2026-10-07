/// A form: the fields a note is mapped to, in display order.
public struct FormSchema: Codable, Equatable, Sendable {
    public var fields: [FormField]

    public init(fields: [FormField]) {
        self.fields = fields
    }

    /// One entity label per field, in field order: `modelLabel` with `description`. The `labelIndex` of an extracted
    /// span is then the index of its field.
    public func extractionSchema() -> GLiNERSchema {
        .entities(fields.map { EntityLabel(name: $0.modelLabel, description: $0.description) })
    }

    /// The `modelLabel` of each option of a list field, in option order, under `modelTask` (the field's name unless
    /// set). There is no label for "not mentioned". A number field has no labels.
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
        /// `modelTask` is the classification task the model reads; nil reads the field's name.
        case list(options: [FormOption], modelTask: String? = nil)
    }

    public var key: String
    /// The name people see. It is also the classification task of a list field without a `modelTask`.
    public var name: String
    /// The entity label the model reads.
    public var modelLabel: String
    public var kind: Kind
    /// The description of `modelLabel` the model reads.
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
    /// The option people see and the answer is compared with.
    public var name: String
    /// The classification label the model reads.
    public var modelLabel: String

    public init(name: String, modelLabel: String? = nil) {
        self.name = name
        self.modelLabel = modelLabel ?? name
    }
}
