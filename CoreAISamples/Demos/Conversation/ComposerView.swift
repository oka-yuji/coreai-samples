import ConversationFeature
import SwiftUI

struct ComposerView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(ConversationHost.self) private var host
    @FocusState private var focused: Bool

    var body: some View {
        @Bindable var chat = chat
        let state = host.state
        let running = state.isRunning || chat.isResponding
        return VStack(alignment: .leading, spacing: 8) {
            if let notice {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.circle")
                    RuntimeMessageText(notice)
                    Spacer(minLength: 0)
                    if host.contextFull || chat.contextFull {
                        Button("New", action: host.startNew)
                    }
                }
                .font(.callout)
                .foregroundStyle(.orange)
            }
            if state.refusedSave && !state.isRunning {
                Button("Continue in Work", systemImage: "arrow.right.circle", action: host.continueInWork)
                    .help("Ask again in Work mode, which can save notes")
            }
            VStack(alignment: .leading, spacing: 4) {
                Picker("Mode", selection: Binding(get: { state.mode }, set: { host.store.send(.modeChanged($0)) })) {
                    Text("Chat").tag(ConversationMode.chat)
                    Text("Work").tag(ConversationMode.work)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .disabled(running)
                Text(
                    state.mode == .chat
                        ? "Conversation. Searches the web when it needs facts." : "Multi-step research and notes."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Message", text: $chat.input, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...6)
                    .padding(8)
                    .background(.background, in: .rect(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.separator))
                    .focused($focused)
                    .disabled(host.contextFull)
                    .onSubmit(submit)
                if running {
                    if state.stopRequested {
                        Text("Stopping…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Button("Stop", systemImage: "stop.circle.fill", action: host.stop)
                        .labelStyle(.iconOnly)
                        .font(.system(size: 26))
                        .buttonStyle(.plain)
                        .foregroundStyle(.tint)
                        .disabled(state.stopRequested)
                        .keyboardShortcut(".", modifiers: .command)
                } else {
                    Button("Send", systemImage: "arrow.up.circle.fill", action: submit)
                        .labelStyle(.iconOnly)
                        .font(.system(size: 26))
                        .buttonStyle(.plain)
                        .foregroundStyle(host.canSend ? Color.accentColor : Color.secondary)
                        .disabled(!host.canSend)
                }
            }
        }
        .padding(10)
    }

    private var notice: String? {
        if host.contextFull { return "The context window is full. Start a new conversation to continue." }
        if chat.ragIsOn, let notice = chat.contextNotice { return notice }
        return host.state.contextEndNotice
    }

    private func submit() {
        guard host.canSend else { return }
        focused = false
        host.submit()
    }
}
