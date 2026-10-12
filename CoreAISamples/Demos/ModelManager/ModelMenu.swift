import AppKit
import SwiftUI

struct ModelMenu: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(ModelsViewModel.self) private var models
    @Environment(DemoNavigator.self) private var navigator
    var agent: AgentViewModel?

    var body: some View {
        let entries = models.menuEntries(current: chat.phase == .ready ? chat.bundleURL : nil)
        Menu {
            if entries.downloaded.isEmpty && entries.onThisMac.isEmpty {
                Button("No models yet") {}
                    .disabled(true)
            }
            if !entries.downloaded.isEmpty {
                Section("Downloaded") { choices(entries.downloaded) }
            }
            if !entries.onThisMac.isEmpty {
                Section("On this Mac") { choices(entries.onThisMac) }
            }
            Divider()
            Button("Manage Models…") { navigator.selection = .models }
        } label: {
            Text(chat.modelName.isEmpty ? "No model loaded" : chat.modelName)
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .fixedSize()
        .disabled(
            Self.isDisabled(
                loading: chat.isLoading, responding: chat.isResponding, busy: chat.externallyBusy,
                agentRunning: agent?.running == true)
        )
        .onReceive(NotificationCenter.default.publisher(for: NSMenu.didBeginTrackingNotification)) { _ in
            models.refresh()
        }
    }

    private func choices(_ entries: [ModelsViewModel.MenuEntry]) -> some View {
        ForEach(entries) { entry in
            Toggle(
                entry.name,
                isOn: Binding(
                    get: { entry.isCurrent },
                    set: { _ in Task { await Self.choose(entry, chat: chat, agent: agent) } }))
        }
    }

    static func isDisabled(loading: Bool, responding: Bool, busy: Bool, agentRunning: Bool) -> Bool {
        loading || responding || busy || agentRunning
    }

    static func choose(_ entry: ModelsViewModel.MenuEntry, chat: ChatViewModel, agent: AgentViewModel?) async {
        guard !entry.isCurrent else { return }
        agent?.reset()
        await chat.load(entry.url)
    }
}
