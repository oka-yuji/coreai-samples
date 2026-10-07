import DecideFeature
import SwiftUI

struct DecideItemEditor: View {
    @Environment(\.dismiss) private var dismiss
    @State var item: DecideItem
    let save: (DecideItem) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Edit checklist item").font(.title2.bold())
            Form {
                TextField("Item name", text: $item.key)
                TextField("Instructions", text: $item.instructions, axis: .vertical)
                    .lineLimit(2...5)
                Picker("Answer type", selection: $item.kind) {
                    ForEach(DecideQuestionKind.allCases, id: \.self) { kind in
                        Text(kind.title).tag(kind)
                    }
                }
                .onChange(of: item.kind) { _, kind in
                    switch kind {
                    case .noul:
                        item.options = [
                            DecideOption(id: 0, key: "true", text: "The answer is yes."),
                            DecideOption(id: 1, key: "false", text: "The answer is no."),
                        ]
                    case .choice:
                        item.options = DecideItem.blank(id: item.id).options
                    case .score:
                        item.options = [
                            DecideOption(id: 0, key: "0", text: "Low"),
                            DecideOption(id: 1, key: "1", text: "Medium"),
                            DecideOption(id: 2, key: "2", text: "High"),
                        ]
                    }
                }
            }
            Text("Allowed options").font(.headline)
            ScrollView {
                VStack(spacing: 10) {
                    ForEach($item.options) { $option in
                        HStack(alignment: .top) {
                            if item.kind == .choice {
                                TextField("Option ID", text: $option.key).frame(width: 120)
                            } else {
                                Text(option.key).font(.caption.monospaced()).frame(width: 50, alignment: .leading)
                            }
                            TextField("Meaning", text: $option.text, axis: .vertical).lineLimit(1...3)
                            if item.kind != .noul {
                                Button("Remove option", systemImage: "minus.circle") {
                                    item.options.removeAll { $0.id == option.id }
                                    renumberScore()
                                }
                                .labelStyle(.iconOnly).buttonStyle(.borderless)
                                .disabled(item.options.count <= 1)
                            }
                        }
                    }
                }
                .padding(2)
            }
            .frame(minHeight: 120, maxHeight: 270)
            if item.kind != .noul {
                Button("Add option", systemImage: "plus") {
                    let id = (item.options.map(\.id).max() ?? -1) + 1
                    item.options.append(
                        DecideOption(id: id, key: item.kind == .score ? "\(item.options.count)" : "option\(id + 1)"))
                }
                .disabled(item.options.count >= 8)
            }
            if let problem { Text(problem).font(.caption).foregroundStyle(.orange) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Save") { save(item) }.buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction).disabled(problem != nil)
            }
        }
        .padding(24)
        .frame(width: 560)
    }

    private var problem: String? {
        DecideInput(document: "validation", items: [item]).problem
    }

    private func renumberScore() {
        guard item.kind == .score else { return }
        for index in item.options.indices { item.options[index].key = String(index) }
    }
}
