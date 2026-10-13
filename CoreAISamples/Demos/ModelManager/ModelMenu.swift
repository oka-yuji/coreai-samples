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
        let name = chat.modelName.isEmpty ? Text("No model loaded") : Text(verbatim: chat.modelName)
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
            name
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: 360, alignment: .leading)
        }
        .fixedSize()
        .accessibilityLabel("Model")
        .accessibilityValue(name)
        .help("Choose the model the conversation uses")
        .disabled(
            Self.isDisabled(
                loading: chat.isLoading, responding: chat.isResponding, busy: chat.externallyBusy,
                running: host.state.isRunning)
        )
        .onReceive(NotificationCenter.default.publisher(for: NSMenu.didBeginTrackingNotification)) { _ in
            models.refresh()
        }
        .task(id: chosen?.id) {
            guard let entry = chosen else { return }
            await Self.choose(entry, chat: chat, host: host)
            chosen = nil
        }
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

    static func choose(_ entry: ModelsViewModel.MenuEntry, chat: ChatViewModel, host: ConversationHost?) async {
        guard !entry.isCurrent else { return }
        host?.startNew()
        await chat.load(entry.url)
    }
}
