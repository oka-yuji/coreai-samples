import AppKit
import SwiftUI

struct ModelMenu: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(ModelsViewModel.self) private var models
    @Environment(DemoNavigator.self) private var navigator
    @Environment(ConversationHost.self) private var host

    @State private var chosen: ModelsViewModel.MenuEntry?

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
            modelLabel
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: 360, alignment: .leading)
        }
        .accessibilityLabel("Model")
        .accessibilityValue(modelLabel)
        .help("Choose the model the conversation uses")
        .disabled(
            Self.isDisabled(
                loading: chat.isLoading, responding: chat.isResponding, busy: chat.externallyBusy,
                running: !host.state.canLeave)
        )
        .onReceive(NotificationCenter.default.publisher(for: NSMenu.didBeginTrackingNotification)) { _ in
            models.refresh()
        }
        .task(id: chosen?.id) {
            guard let entry = chosen else { return }
            await ModelSwitch.choose(entry.url, chat: chat, host: host)
            chosen = nil
        }
    }

    private var modelLabel: Text {
        chat.modelName.isEmpty ? Text("Choose model") : Text(verbatim: chat.modelName)
    }

    private func choices(_ entries: [ModelsViewModel.MenuEntry]) -> some View {
        ForEach(entries) { entry in
            Toggle(
                entry.name,
                isOn: Binding(
                    get: { entry.isCurrent },
                    set: { _ in chosen = entry }))
        }
    }

    static func isDisabled(loading: Bool, responding: Bool, busy: Bool, running: Bool) -> Bool {
        loading || responding || busy || running
    }
}

@MainActor
enum ModelSwitch {
    static func choose(_ url: URL, chat: ChatViewModel, host: ConversationHost) async -> Bool {
        guard
            !ModelMenu.isDisabled(
                loading: chat.isLoading, responding: chat.isResponding, busy: chat.externallyBusy,
                running: host.state.isRunning),
            !isLoaded(url, phase: chat.phase, current: chat.bundleURL)
        else { return false }
        guard host.prepareModelSwitch(to: url) else { return false }
        await chat.load(url)
        return true
    }

    static func isLoaded(_ url: URL, phase: ChatViewModel.Phase, current: URL?) -> Bool {
        phase == .ready && current.map(ModelStorage.key) == ModelStorage.key(url)
    }
}
