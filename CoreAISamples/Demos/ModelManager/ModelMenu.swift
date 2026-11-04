import AppKit
import SwiftUI

struct ModelMenu: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(MacModelLibrary.self) private var library
    @Environment(DemoNavigator.self) private var navigator
    @Environment(ConversationHost.self) private var host

    @Environment(MacModelChoice.self) private var choice

    var body: some View {
        Menu {
            if let resource = library.installed(.qwen) {
                Toggle(
                    "Qwen3.8 27B",
                    isOn: Binding(
                        get: { ModelSwitch.isLoaded(resource.directory, phase: chat.phase, current: chat.bundleURL) },
                        set: { _ in
                            guard !ModelSwitch.isLoaded(resource.directory, phase: chat.phase, current: chat.bundleURL)
                            else { return }
                            if library.resource(inUse: .qwen) == nil && !choice.state.selecting { library.use(.qwen) }
                            choice.send(.choose)
                        }))
            } else {
                Button("No models yet") {}.disabled(true)
            }
            if chat.phase == .ready {
                Button("Unload model") { choice.send(.unload) }
            }
            Divider()
            Button("Manage Models…") { navigator.selection = .models }
        } label: {
            modelLabel
                .font(.body.weight(.medium))
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .accessibilityLabel("Model")
        .accessibilityValue(modelLabel)
        .help("Choose the model the conversation uses")
        .disabled(
            Self.isDisabled(
                loading: chat.isLoading, responding: chat.isResponding, busy: chat.externallyBusy,
                running: !host.state.canLeave || choice.state.selecting || choice.state.unloading)
        )
        .onReceive(NotificationCenter.default.publisher(for: NSMenu.didBeginTrackingNotification)) { _ in
            library.refresh()
        }

    }

    private var modelLabel: Text {
        chat.modelName.isEmpty ? Text("Choose model") : Text(verbatim: chat.modelName)
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
