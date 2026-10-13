import AppKit
import SwiftUI

struct ConversationSettingsView: View {
    @Environment(\.locale) private var locale
    @Environment(ChatViewModel.self) private var chat
    @Environment(ConversationHost.self) private var host

    var body: some View {
        @Bindable var preferences = host.preferences
        return VStack(alignment: .leading, spacing: 8) {
            Picker("Reasoning effort", selection: $preferences.effort) {
                ForEach(AgentReasoningEffort.allCases) { Text(LocalizedStringKey($0.label)).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 320)
            HStack(spacing: 8) {
                Text("Max steps (Work): \(preferences.maxSteps)")
                    .monospacedDigit()
                    .frame(width: 190, alignment: .leading)
                Slider(value: sliderBinding($preferences.maxSteps), in: 1...16, step: 1)
                    .frame(maxWidth: 320)
                    .accessibilityLabel("Max steps (Work)")
            }
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
            HStack(spacing: 6) {
                Text("Brave Search API key")
                SecureField(
                    "Brave Search API key", text: $preferences.braveAPIKey,
                    prompt: Text("optional — empty uses DuckDuckGo")
                )
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 280)
            }
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
