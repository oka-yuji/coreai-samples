import SwiftUI

struct ConversationSidebar: View {
    @Environment(DemoNavigator.self) private var navigator
    @Environment(ModelsViewModel.self) private var models
    @State private var collapsed: Set<DemoModel> = []
    @State private var manageExpanded = true

    var body: some View {
        VStack(spacing: 0) {
            storage
            List(selection: selection) {
                ForEach(DemoModel.allCases, id: \.self) { model in
                    Section(isExpanded: expanded(model)) {
                        ForEach(model.demos) { demoRow($0) }
                    } header: {
                        Text(verbatim: model.title)
                    }
                }
                Section("Manage", isExpanded: $manageExpanded) { demoRow(.models) }
            }
        }
        .demoNavigationTitle("Models")
        .navigationSplitViewColumnWidth(min: 220, ideal: 260)
        .onAppear { models.refresh() }
    }

    private var selection: Binding<Demo?> {
        Binding {
            navigator.selection
        } set: {
            if let demo = $0 { navigator.selection = demo }
        }
    }

    private var storage: some View {
        let downloaded = models.downloadedSize.formatted(.byteCount(style: .file))
        let available = models.availableSpace.formatted(.byteCount(style: .file))
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

    private func demoRow(_ demo: Demo) -> some View {
        Label(LocalizedStringKey(demo.summary), systemImage: demo.systemImage).lineLimit(nil).tag(demo)
    }
}
