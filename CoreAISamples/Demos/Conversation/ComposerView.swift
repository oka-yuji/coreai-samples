import ConversationFeature
import SwiftUI

struct ComposerView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(ConversationHost.self) private var host

    var body: some View {
        @Bindable var chat = chat
        let state = host.state
        let running = state.isRunning || chat.isResponding
        return VStack(alignment: .leading, spacing: 8) {
            if let notice {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.circle")
                    notice
                    Spacer(minLength: 0)
                    if host.contextFull {
                        Button("New", action: host.startNew)
                    }
                }
                .font(.callout)
                .foregroundStyle(.orange)
            }
            if case .failed(let detail) = state.persistence {
                Label {
                    Text("The conversation could not be saved: \(detail)")
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .font(.callout)
                .foregroundStyle(.orange)
            }
            if ConversationFeature.continueEligibility(state) {
                Button("Continue answer", systemImage: "arrow.turn.down.right", action: host.continueAnswer)
                    .help("Generate the rest as a new turn without repeating the answer")
                    .disabled(!host.canContinueAnswer)
            }
            if state.refusedSave && !state.isRunning && !host.contextFull {
                Button("Continue in Work", systemImage: "arrow.right.circle", action: host.continueInWork)
                    .help("Ask again in Work mode, which can save notes")
                    .disabled(!host.canContinueInWork)
            }
            VStack(alignment: .leading, spacing: 4) {
                Picker("Mode", selection: Binding(get: { state.mode }, set: { host.store.send(.modeChanged($0)) })) {
                    Text("Chat").tag(ConversationMode.chat)
                    Text("Work").tag(ConversationMode.work)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .disabled(running || chat.ragIsOn)
                Text(
                    chat.ragIsOn
                        ? "Answers from the collected sources in one reply."
                        : state.mode == .chat
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

    private var notice: Text? {
        if host.contextFull { return Text("The context window is full. Start a new conversation to continue.") }
        guard let reserved = host.state.contextEndReserved else { return nil }
        return Text(
            "The reply stopped at the end of the context window (\(reserved) tokens were left for it). Start a new conversation if the next message does not fit."
        )
    }

    private func submit() {
        guard host.canSend else { return }
        host.submit()
    }
}
