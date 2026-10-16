import ConversationFeature
import SwiftUI

struct ConversationSidebar: View {
    private enum Selection: Hashable {
        case conversation(String)
        case tool(Demo)
        case new
    }

    @Environment(ConversationHost.self) private var host
    @Environment(DemoNavigator.self) private var navigator
    @State private var editingID: String?
    @State private var editingTitle = ""

    var body: some View {
        VStack(spacing: 0) {
            List(selection: selection) {
                Section("Conversations") {
                    if host.state.turns.isEmpty && host.state.readOnlyMarkdown == nil {
                        Label("New conversation", systemImage: "bubble.left").tag(Selection.new)
                    }
                    ForEach(host.state.conversationSummaries) { item in
                        row(item)
                            .tag(Selection.conversation(item.id))
                            .disabled(!host.canManageHistory && item.id != host.state.id)
                            .contextMenu {
                                Button("Rename…") {
                                    editingID = item.id
                                    editingTitle = item.title
                                }.disabled(item.revision == nil || !host.canManageHistory)
                                Button("Delete…", role: .destructive) { host.requestDeleteConversation(item.id) }
                                    .disabled(
                                        !host.canManageHistory
                                            || !host.state.library.items.contains { $0.id == item.id })
                            }
                    }
                }
                Section("Tools") {
                    ForEach(Demo.allCases.filter { $0 != .chat }) { demo in
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(LocalizedStringKey(demo.title))
                                Text(LocalizedStringKey(demo.summary)).font(.caption).foregroundStyle(.secondary)
                            }
                        } icon: {
                            Image(systemName: demo.systemImage)
                        }
                        .tag(Selection.tool(demo))
                    }
                }
            }
            if host.state.library.loading != nil || host.state.library.isMutating {
                ProgressView().controlSize(.small).padding(8)
            }
            if host.state.library.error != nil || host.state.library.exportRetryID != nil {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Could not update conversations").font(.caption.bold())
                    if let message = host.state.library.error {
                        Text(message).font(.caption).textSelection(.enabled)
                    } else {
                        Text("Markdown export is not up to date.").font(.caption)
                    }
                    if host.state.library.exportRetryID != nil {
                        Button("Retry save", action: host.retryConversationExport).disabled(!host.canManageHistory)
                    }
                    Button("Refresh") { host.store.send(.library(.refresh)) }
                        .disabled(host.state.library.isMutating)
                }.padding(8)
            }
        }
        .demoNavigationTitle("Demos")
        .navigationSplitViewColumnWidth(min: 220, ideal: 260)
        .alert(
            "Rename conversation", isPresented: Binding(get: { editingID != nil }, set: { if !$0 { editingID = nil } })
        ) {
            TextField("Title", text: $editingTitle)
            Button("Save") {
                if let id = editingID { host.renameConversation(id, title: editingTitle) }
                editingID = nil
            }.disabled(editingTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            Button("Cancel", role: .cancel) { editingID = nil }
        }
        .confirmationDialog(
            "Delete conversation?",
            isPresented: Binding(
                get: { host.state.library.pendingDelete != nil },
                set: { if !$0 { host.store.send(.library(.cancelDelete)) } }), titleVisibility: .visible
        ) {
            Button("Move to Trash", role: .destructive, action: host.confirmDeleteConversation)
            Button("Cancel", role: .cancel) { host.store.send(.library(.cancelDelete)) }
        } message: {
            Text("The conversation and its Markdown export will be moved to the Trash.")
        }
    }

    private var selection: Binding<Selection?> {
        Binding {
            guard let demo = navigator.selection else { return nil }
            if demo != .chat { return .tool(demo) }
            return host.state.turns.isEmpty && host.state.readOnlyMarkdown == nil ? .new : .conversation(host.state.id)
        } set: { value in
            switch value {
            case .conversation(let id):
                if id == host.state.id || host.openConversation(id) { navigator.selection = .chat }
            case .tool(let demo): navigator.selection = demo
            case .new: navigator.selection = .chat
            case nil: break
            }
        }
    }

    private func row(_ item: ConversationSummary) -> some View {
        Label {
            VStack(alignment: .leading, spacing: 3) {
                Text(item.title.isEmpty ? item.id : item.title).lineLimit(2)
                HStack(spacing: 5) {
                    if let mode = item.mode { Text(LocalizedStringKey(mode == .chat ? "Chat" : "Work")) }
                    Text(item.updatedAt, format: .dateTime.month(.abbreviated).day().hour().minute())
                }.font(.caption).foregroundStyle(.secondary)
            }
        } icon: {
            Image(
                systemName: item.revision == nil && (item.id != host.state.id || host.state.readOnlyMarkdown != nil)
                    ? "doc.text" : "bubble.left.and.bubble.right")
        }
    }
}
