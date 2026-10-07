import Foundation
import GLiNERCore

/// What people type to add or change a form field.
public struct FieldDraft: Equatable, Sendable {
    public enum Kind: String, CaseIterable, Sendable {
        case number
        case list
    }

    /// The field being changed; nil for a new field.
    public var key: String?
    public var name: String
    /// The entity label the model reads; empty means the name.
    public var modelLabel: String
    public var kind: Kind
    public var unit: String
    /// Option names, separated by commas (ASCII or ideographic) or line breaks.
    public var options: String
    public var description: String

    /// A draft of `field`, or an empty number field.
    public init(field: FormField? = nil) {
        key = field?.key
        name = field?.name ?? ""
        modelLabel = field?.modelLabel ?? ""
        description = field?.description ?? ""
        switch field?.kind {
        case .list(let options, _)?:
            kind = .list
            unit = ""
            self.options = options.map(\.name).joined(separator: ", ")
        case .number(let unit)?:
            kind = .number
            self.unit = unit
            options = ""
        case nil:
            kind = .number
            unit = ""
            options = ""
        }
    }

    /// The field this draft makes in `form`, or nil when the name is empty or a list has no option (or more than the
    /// model takes). Options that keep their name keep their model label, and a list keeps its task while the name
    /// stays. A new field gets the first free key of `field1`, `field2`, ...
    public func field(in form: FormSchema) -> FormField? {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        let old = form.fields.first { $0.key == key }
        let label = modelLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        let note = description.trimmingCharacters(in: .whitespacesAndNewlines)
        let fieldKind: FormField.Kind
        switch kind {
        case .number:
            fieldKind = .number(unit: unit.trimmingCharacters(in: .whitespacesAndNewlines))
        case .list:
            let names = options.components(separatedBy: CharacterSet(charactersIn: ",\u{3001}\u{FF0C}\n"))
                .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            guard !names.isEmpty, names.count <= PreparedInput.labelCapacity else { return nil }
            var oldOptions: [FormOption] = []
            var oldTask: String?
            if case .list(let options, let task)? = old?.kind {
                oldOptions = options
                oldTask = old?.name == name ? task : nil
            }
            let options = names.map { name in oldOptions.first { $0.name == name } ?? FormOption(name: name) }
            fieldKind = .list(options: options, modelTask: oldTask)
        }
        let keys = Set(form.fields.map(\.key))
        let newKey = (1...).lazy.map { "field\($0)" }.first { !keys.contains($0) } ?? "field"
        return FormField(
            key: old?.key ?? newKey, name: name, modelLabel: label.isEmpty ? name : label, kind: fieldKind,
            description: note.isEmpty ? nil : note)
    }
}
