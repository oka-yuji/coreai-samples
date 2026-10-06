import ExtractFeature
import GLiNERCore
import SwiftUI

/// Adds a field to the form, or changes or removes one. `save` gets the new form and the field to select.
struct FieldEditor: View {
    @State var draft: FieldDraft
    let form: FormSchema
    let save: (FormSchema, String?) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(draft.key == nil ? "Add Field" : "Edit Field").font(.title3.bold())
            Form {
                TextField("Name", text: $draft.name)
                TextField("Label for the model", text: $draft.modelLabel, prompt: Text("Same as the name"))
                Picker("Type", selection: $draft.kind) {
                    Text("Number").tag(FieldDraft.Kind.number)
                    Text("List").tag(FieldDraft.Kind.list)
                }
                .pickerStyle(.segmented)
                if draft.kind == .number {
                    TextField("Unit", text: $draft.unit, prompt: Text("MPa"))
                } else {
                    TextField("Options", text: $draft.options, prompt: Text("Comma-separated"))
                }
                TextField("Description", text: $draft.description, prompt: Text("Optional"), axis: .vertical)
                    .lineLimit(2...4)
            }
            HStack {
                if let key = draft.key {
                    Button("Remove Field", role: .destructive) {
                        save(FormSchema(fields: form.fields.filter { $0.key != key }), nil)
                        dismiss()
                    }
                }
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") {
                    guard let field = draft.field(in: form) else { return }
                    var fields = form.fields
                    if let index = fields.firstIndex(where: { $0.key == field.key }) {
                        fields[index] = field
                    } else {
                        fields.append(field)
                    }
                    save(FormSchema(fields: fields), field.key)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(draft.field(in: form) == nil)
            }
        }
        .padding(20)
        .frame(width: 440)
    }
}
