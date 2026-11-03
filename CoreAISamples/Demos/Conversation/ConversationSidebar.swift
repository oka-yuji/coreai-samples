import FeatureStore
import ModelLibraryFeature
import ModelLibraryUI
import SwiftUI

struct ConversationSidebar: View {
    @Environment(DemoNavigator.self) private var navigator
    @Environment(MacModelLibrary.self) private var library
    @State private var collapsed: Set<DemoModel> = []
    @State private var manageExpanded = true
    @State private var deletion: ModelDeletionRequest?

    var body: some View {
        VStack(spacing: 0) {
            storage
            list
        }
        .demoNavigationTitle("Models")
        .navigationSplitViewColumnWidth(min: 220, ideal: 260)
        .onAppear { library.refresh() }
        .onChange(of: settledArtifacts) { library.measureStorage() }
    }

    @ViewBuilder private var list: some View {
        let list = List(selection: selection) {
            ForEach(DemoModel.allCases, id: \.self) { model in
                Section(isExpanded: expanded(model)) {
                    if let failure = failure(of: model) { LibraryFailureView(failure: failure) }
                    ForEach(model.demos) { demoRow($0) }
                } header: {
                    header(model)
                }
            }
            Section("Manage", isExpanded: $manageExpanded) { demoRow(.models) }
        }
        if let store = library.store {
            list.modelDeletionAlert(store: store, deletion: $deletion)
        } else {
            list
        }
    }

    private var selection: Binding<Demo?> {
        Binding {
            navigator.selection
        } set: {
            if let demo = $0 { navigator.selection = demo }
        }
    }

    private var settledArtifacts: [String] {
        library.store?.state.installations.filter { !$0.value.phase.isBusy }.keys.sorted() ?? []
    }

    private var storage: some View {
        let downloaded = library.downloadedSize.formatted(.byteCount(style: .file))
        let available = library.availableSpace.formatted(.byteCount(style: .file))
        return Text("Downloaded \(downloaded) · Available \(available)")
            .font(.caption)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
    }

    private func expanded(_ model: DemoModel) -> Binding<Bool> {
        Binding {
            !collapsed.contains(model)
        } set: { isExpanded in
            if isExpanded { collapsed.remove(model) } else { collapsed.insert(model) }
        }
    }

    private func header(_ model: DemoModel) -> some View {
        HStack(spacing: 0) {
            Text(verbatim: model.title)
            Spacer(minLength: 4)
            if let store = library.store, let entry = store.state.catalog.model(model.libraryID) {
                ModelInformationButton(store: store, model: entry)
                if store.state.selectedArtifact(for: entry.id)?.sourceKind != nil {
                    ModelAcquisitionControls(store: store, model: entry) {
                        deletion = ModelDeletionRequest(model: entry, artifact: $0)
                    }
                }
            }
        }
        .controlSize(.small)
    }

    private func failure(of model: DemoModel) -> ModelLibraryFailure? {
        guard let state = library.store?.state, let artifact = state.selectedArtifact(for: model.libraryID) else {
            return nil
        }
        return state.installations[artifact.id]?.failure
    }

    private func demoRow(_ demo: Demo) -> some View {
        Label(LocalizedStringKey(demo.summary), systemImage: demo.systemImage).lineLimit(nil).tag(demo)
    }
}

private struct ModelInformationButton: View {
    let store: Store<ModelLibraryFeature>
    let model: LibraryModel
    @State private var showing = false
    @State private var message: Text?
    @Environment(\.locale) private var locale

    var body: some View {
        Button {
            message = nil
            showing = true
        } label: {
            Image(systemName: "info.circle")
                .frame(minWidth: 44, minHeight: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Model information")
        .accessibilityValue(Text(verbatim: model.name))
        .popover(isPresented: $showing) {
            VStack(alignment: .leading, spacing: 12) {
                ModelInformationView(store: store, model: model)
                if let artifact = store.state.selectedArtifact(for: model.id),
                    let resource = store.state.installations[artifact.id]?.resource
                {
                    Button("Clear Prepared Copy") {
                        do {
                            try MacModelLibrary.clearPrepared(resource.directory, artifact: artifact)
                            message = Text(
                                "Removed the prepared copy of \(model.name). The next load prepares it again.")
                        } catch {
                            message = Text(
                                "Could not remove the prepared copy of \(model.name): \(error.localizedDescription)")
                        }
                    }
                    .help("Remove the copy this app prepared for the GPU. The next load prepares it again.")
                    .disabled(store.state.session?.selection.modelID == model.id)
                    if let message { message.font(.callout).foregroundStyle(.secondary) }
                }
            }
            .padding()
            .frame(width: 360)
            .environment(\.locale, locale)
        }
    }
}
