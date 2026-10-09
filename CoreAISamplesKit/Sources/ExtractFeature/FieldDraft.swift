import Foundation
import GLiNERCore

public struct FieldDraft: Equatable, Sendable {
    public enum Kind: String, CaseIterable, Sendable {
        case number
        case list
    }

    public var key: String?
    public var name: String

    public var modelLabel: String
    public var kind: Kind
    public var unit: String

    public var options: String
    public var description: String

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
