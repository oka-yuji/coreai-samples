import AppKit
import SwiftUI

struct ModelMenu: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(ModelsViewModel.self) private var models
    @Environment(DemoNavigator.self) private var navigator
    @Environment(AgentViewModel.self) private var agent

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
                .frame(maxWidth: 360, alignment: .leading)
        }
        .fixedSize()
        .accessibilityLabel("Model")
        .accessibilityValue(chat.modelName.isEmpty ? "No model loaded" : chat.modelName)
        .help("Choose the model Chat and Agent use")
        .disabled(
            Self.isDisabled(
                loading: chat.isLoading, responding: chat.isResponding, busy: chat.externallyBusy,
                agentRunning: agent.running)
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
                    set: { _ in Task { await ModelSwitch.choose(entry.url, chat: chat, agent: agent) } }))
        }
    }

    static func isDisabled(loading: Bool, responding: Bool, busy: Bool, agentRunning: Bool) -> Bool {
        loading || responding || busy || agentRunning
    }
}

@MainActor
enum ModelSwitch {
    static func choose(_ url: URL, chat: ChatViewModel, agent: AgentViewModel) async -> Bool {
        guard
            !ModelMenu.isDisabled(
                loading: chat.isLoading, responding: chat.isResponding, busy: chat.externallyBusy,
                agentRunning: agent.running),
            !isLoaded(url, phase: chat.phase, current: chat.bundleURL)
        else { return false }
        agent.reset()
        await chat.load(url)
        return true
    }

    static func isLoaded(_ url: URL, phase: ChatViewModel.Phase, current: URL?) -> Bool {
        phase == .ready && current.map(ModelStorage.key) == ModelStorage.key(url)
    }
}
