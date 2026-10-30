import AppKit
import ConversationFeature
import FeatureStore
import SwiftUI

struct ConversationSettingsView: View {
    @Environment(\.locale) private var locale
    @Environment(ChatViewModel.self) private var chat
    @Environment(ConversationHost.self) private var host

    var body: some View {
        @Bindable var preferences = host.preferences
        let steps = Binding(
            get: { host.state.mode == .chat ? preferences.chatMaxSteps : preferences.maxSteps },
            set: { if host.state.mode == .chat { preferences.chatMaxSteps = $0 } else { preferences.maxSteps = $0 } })
        return VStack(alignment: .leading, spacing: 8) {
            Picker("Reasoning effort", selection: $preferences.effort) {
                ForEach(AgentReasoningEffort.allCases) { Text(LocalizedStringKey($0.label)).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 320)
            HStack(spacing: 8) {
                Text("Max steps per reply: \(steps.wrappedValue)")
                    .monospacedDigit()
                    .frame(width: 190, alignment: .leading)
                Slider(value: sliderBinding(steps), in: 1...Double(TurnPolicy.maximumSteps), step: 1)
                    .frame(maxWidth: 320)
                    .accessibilityLabel("Max steps per reply")
            }
            .help("Applies to the selected mode. A final answer without tools gets one extra step at the limit.")
            HStack(spacing: 8) {
                Text("Page budget: \(preferences.pageBudget) chars")
                    .monospacedDigit()
                    .frame(width: 190, alignment: .leading)
                Slider(value: sliderBinding($preferences.pageBudget), in: 500...20_000, step: 500)
                    .frame(maxWidth: 320)
                    .accessibilityLabel("Page budget")
            }
            Picker("Max tokens", selection: $preferences.maxTokens) {
                Text("Default (Chat 2048, Work 4096)").tag(Int?.none)
                ForEach(ConversationPreferences.maxTokenChoices, id: \.self) { Text("\($0)").tag(Int?.some($0)) }
            }
            .fixedSize()
            searchKeyField(preferences.searchKey)
            Toggle(
                "Allow file operations (move_note, from the next conversation)", isOn: $preferences.fileOperations)
            Toggle("Citation guard", isOn: $preferences.citationGuard)
                .help(
                    "In Work, check the answer's and the notes' URLs against the pages fetched in this conversation, ask the model to fix them once, and keep notes as drafts until the check passes"
                )
            HStack(spacing: 6) {
                Text("Notes folder")
                Text(preferences.notesDirectory.path(percentEncoded: false))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.head)
                Button("Choose…") { chooseNotesFolder() }
                Button("Default") { preferences.notesFolderPath = "" }
                    .disabled(preferences.notesFolderPath.isEmpty)
            }
            RAGControls()
        }
        .controlSize(.small)
        .font(.caption)
        .disabled(host.state.isRunning || chat.isResponding)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func searchKeyField(_ searchKey: Store<SearchKeyFeature>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text("Brave Search API key")
                SecureField(
                    "Brave Search API key",
                    text: Binding(get: { searchKey.state.draft }, set: { searchKey.send(.draftChanged($0)) }),
                    prompt: Text("optional — empty uses DuckDuckGo")
                )
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 280)
                .disabled(!searchKey.state.canEdit)
                .onSubmit { searchKey.send(.saveRequested) }
                Button("Save") { searchKey.send(.saveRequested) }
                    .disabled(!searchKey.state.canSave)
            }
            searchKeyStatus(searchKey)
        }
    }

    @ViewBuilder
    private func searchKeyStatus(_ searchKey: Store<SearchKeyFeature>) -> some View {
        switch searchKey.state.status {
        case .reading: Text("Reading the saved key…").foregroundStyle(.secondary)
        case .saving: Text("Saving the key…").foregroundStyle(.secondary)
        case .unsaved: Text("Not saved").foregroundStyle(.secondary)
        case .saved: Text("Saved in the keychain").foregroundStyle(.secondary)
        case .empty: EmptyView()
        case .failed(let failure):
            HStack(spacing: 6) {
                failureText(failure).foregroundStyle(.red)
                Button("Retry") { if let retry = searchKey.state.retry { searchKey.send(retry) } }
                    .disabled(searchKey.state.retry == nil)
            }
        }
    }

    private func failureText(_ failure: SearchKeyFeature.Failure) -> Text {
        switch failure.operation {
        case .read: Text("Could not read the saved key, so searches use DuckDuckGo: \(failure.reason)")
        case .move: Text("Could not move the key to the keychain, so it stays in the settings file: \(failure.reason)")
        case .write: Text("Could not save the key, so the previous key stays in use: \(failure.reason)")
        case .delete: Text("Could not remove the saved key, so it stays in use: \(failure.reason)")
        }
    }

    private func sliderBinding(_ value: Binding<Int>) -> Binding<Double> {
        Binding(get: { Double(value.wrappedValue) }, set: { value.wrappedValue = Int($0.rounded()) })
    }

    private func chooseNotesFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = DemoStrings(locale: locale).text("Choose")
        panel.message = DemoStrings(locale: locale).text("Where Work saves its notes")
        panel.directoryURL = host.preferences.notesDirectory
        guard panel.runModal() == .OK, let url = panel.url else { return }
        host.preferences.notesFolderPath = url.path(percentEncoded: false)
    }
}
