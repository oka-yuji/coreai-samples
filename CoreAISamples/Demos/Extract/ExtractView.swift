import ExtractFeature
import FeatureStore
import Foundation
import GLiNERCore
import SwiftUI
import UniformTypeIdentifiers

struct ExtractView: View {
    let store: Store<ExtractFeature>
    @State private var choosingAsset = false
    @State private var draft: FieldDraft?
    @State private var nextExample = 0

    private var state: ExtractState { store.state }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HSplitView {
                notePane.frame(minWidth: 340, idealWidth: 460, maxWidth: .infinity, maxHeight: .infinity)
                fieldsPane.frame(minWidth: 380, idealWidth: 520, maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .demoNavigationTitle("GLiNER2.5-multi")
        .fileImporter(isPresented: $choosingAsset, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result {
                store.send(.assetFolderChosen(path: url.path(percentEncoded: false)))
            }
        }
        .fileDialogDefaultDirectory(ModelStorage.root)
        .sheet(isPresented: Binding(get: { draft != nil }, set: { if !$0 { draft = nil } })) {
            if let draft {
                FieldEditor(draft: draft, form: state.schema, isLabel: state.mode == .labels) { form, selected in
                    store.send(.fieldsChanged(form))
                    store.send(.select(field: selected))
                }
            }
        }
        .onAppear { store.send(.appeared) }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Button("Open Asset Folder…") { choosingAsset = true }
                assetStatus
                Spacer()
                Button("Paste Example") {
                    let notes = ExtractResources.examples.notes
                    guard !notes.isEmpty else { return }
                    store.send(.textChanged(notes[nextExample % notes.count]))
                    nextExample += 1
                }
                Button("Export JSON") { store.send(.exportRequested) }
                    .disabled(state.text.isEmpty)
            }
            HStack(spacing: 16) {
                Toggle(
                    "Experiment: inspection form",
                    isOn: Binding(get: { state.mode == .form }, set: { store.send(.modeChanged($0 ? .form : .labels)) })
                )
                if state.mode == .form {
                    Picker(
                        "Candidate filter",
                        selection: Binding(
                            get: { state.candidateFilter }, set: { store.send(.candidateFilterChanged($0)) })
                    ) {
                        Text("R1+N1 (clauseFieldNameAndEquipmentNumber)")
                            .tag(CandidateFilter.clauseFieldNameAndEquipmentNumber)
                        Text("R1 (clauseFieldName)").tag(CandidateFilter.clauseFieldName)
                        Text("None").tag(CandidateFilter.none)
                    }
                    .fixedSize()
                }
                Spacer()
            }
            Label(LocalizedStringKey(ExtractRunner.notice), systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(spacing: 10) {
                runStatus
                Spacer()
                exportStatus
            }
            .font(.caption)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    @ViewBuilder private var assetStatus: some View {
        switch state.asset {
        case .none:
            Text("No asset folder").foregroundStyle(.secondary)
        case .opening(let path):
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Opening \(URL(filePath: path).lastPathComponent)…")
            }
        case .ready(let info):
            let seconds = info.loadSeconds.formatted(.number.precision(.fractionLength(1)))
            Text("\(URL(filePath: info.path).lastPathComponent) · loaded in \(seconds) s")
            if let revision = info.revision { Text("Revision: \(String(revision.prefix(7)))") }
        case .failed(_, let message):
            Text("Could not open the asset: \(message)").foregroundStyle(.red)
        }
    }

    @ViewBuilder private var runStatus: some View {
        switch state.phase {
        case .extracting:
            Text("Extracting…").foregroundStyle(.secondary)
        case .failed(let message):
            RuntimeMessageText(message).foregroundStyle(.red)
        case .ready, .idle:
            if let timing = state.result?.timing {
                RuntimeMessageText(ExtractRunner.describe(timing)).monospacedDigit().foregroundStyle(.secondary)
            } else if state.schema.fields.isEmpty {
                Text(LocalizedStringKey(state.mode == .labels ? "Add a label to extract." : "Add a field to extract."))
                    .foregroundStyle(.secondary)
            } else if case .ready = state.asset {
                Text(
                    LocalizedStringKey(state.text.isEmpty ? "Paste or type a note." : "Waiting for the text to settle…")
                )
                .foregroundStyle(.secondary)
            } else {
                Text("Open an asset folder to extract.").foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var exportStatus: some View {
        switch state.export {
        case .none:
            EmptyView()
        case .saved(let path):
            Button("Saved \(URL(filePath: path).lastPathComponent)") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(filePath: path)])
            }
            .buttonStyle(.link)
        case .failed(let message):
            Text("Export failed: \(message)").foregroundStyle(.red)
        }
    }

    private var notePane: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Note").font(.headline)
            NoteEditor(
                text: state.text, onChange: { store.send(.textChanged($0)) },
                onComposing: { store.send(.compositionChanged($0)) }
            )
            .frame(minHeight: 110, idealHeight: 150, maxHeight: 220)
            .overlay(RoundedRectangle(cornerRadius: 4).stroke(.separator))
            HStack {
                Text("Evidence").font(.headline)
                Spacer()
                if state.result?.mapping.corrections.isEmpty == false {
                    Label("Correction cue (rule)", systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            ScrollView {
                Text(highlighted)
                    .textSelection(.enabled)
                    .lineSpacing(4)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .background(RoundedRectangle(cornerRadius: 4).fill(Color(nsColor: .textBackgroundColor)))
            .overlay(RoundedRectangle(cornerRadius: 4).stroke(.separator))
            legend
        }
        .padding(14)
    }

    private var legend: some View {
        let fields = Array(state.schema.fields.enumerated())
        return ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(fields, id: \.element.key) { index, field in
                    Button {
                        store.send(.select(field: state.selectedField == field.key ? nil : field.key))
                    } label: {
                        HStack(spacing: 4) {
                            RoundedRectangle(cornerRadius: 2).fill(Self.color(index)).frame(width: 10, height: 10)
                            Text(field.name).fontWeight(state.selectedField == field.key ? .bold : .regular)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
            .font(.caption)
        }
    }

    private var highlighted: AttributedString {
        let text = state.text
        var attributed = AttributedString(text)
        guard let mapping = state.result?.mapping else { return attributed }
        func range(_ scalars: Range<Int>) -> Range<AttributedString.Index>? {
            guard let utf16 = text.utf16Range(fromScalarRange: scalars) else { return nil }
            return Range(NSRange(location: utf16.lowerBound, length: utf16.count), in: attributed)
        }
        let order = mapping.fields.indices.sorted {
            (mapping.fields[$0].field.key == state.selectedField ? 1 : 0, $0)
                < (mapping.fields[$1].field.key == state.selectedField ? 1 : 0, $1)
        }
        for index in order {
            let selected = mapping.fields[index].field.key == state.selectedField
            for candidate in mapping.fields[index].candidates {
                guard let range = range(candidate.span.scalarRange) else { continue }
                attributed[range].backgroundColor = Self.color(index).opacity(selected ? 0.55 : 0.22)
                if selected {
                    attributed[range].underlineStyle = .single
                    attributed[range].inlinePresentationIntent = .stronglyEmphasized
                }
            }
        }
        for cue in mapping.corrections {
            guard let range = range(cue.range) else { continue }
            attributed[range].foregroundColor = .red
            attributed[range].underlineStyle = Text.LineStyle(pattern: .dot, color: .red)
        }
        return attributed
    }

    private var fieldsPane: some View {
        let isLabel = state.mode == .labels
        return VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(LocalizedStringKey(isLabel ? "Labels" : "Form")).font(.headline)
                Spacer()
                if !isLabel {
                    Button("Reset Form") { store.send(.fieldsChanged(ExtractResources.form)) }
                        .disabled(state.form == ExtractResources.form)
                }
                Button(LocalizedStringKey(isLabel ? "Add Label" : "Add Field")) { draft = FieldDraft() }
                    .disabled(state.schema.fields.count >= PreparedInput.labelCapacity)
            }
            .padding(14)
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(Array(state.schema.fields.enumerated()), id: \.element.key) { index, field in
                        FieldRow(
                            field: field, isLabel: isLabel, color: Self.color(index),
                            mapping: state.result?.mapping.fields.first { $0.field.key == field.key },
                            confirmation: state.confirmations[field.key], cues: state.cues(forField: field.key),
                            selected: state.selectedField == field.key, send: store.send,
                            edit: { draft = FieldDraft(field: field) })
                    }
                }
                .padding([.horizontal, .bottom], 14)
            }
        }
    }

    static func color(_ index: Int) -> Color {
        let palette: [Color] = [
            .orange, .blue, .green, .pink, .purple, .teal, .yellow, .red, .mint, .indigo, .brown, .cyan,
        ]
        return palette[index % palette.count]
    }
}
