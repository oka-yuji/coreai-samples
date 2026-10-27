#if os(iOS)
    import SpeechFeature
    import SwiftUI

    struct PhoneSpeechContent: View {
        let state: SpeechState
        let send: (SpeechAction) -> Void
        let isPermitted: @MainActor () -> Bool
        @Environment(\.locale) private var locale
        private enum Field: Hashable { case text, seed }
        @FocusState private var editing: Field?

        var body: some View {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text("Text").font(.headline)
                    Spacer()
                    Menu("Examples") {
                        ForEach(state.examples) { example in
                            Button(LocalizedStringKey("Speech example: " + example.title)) {
                                perform(.setText(example.text))
                            }
                        }
                    }
                    .disabled(!editable)
                }
                TextEditor(text: Binding(get: { state.text }, set: { perform(.setText($0)) }))
                    .frame(minHeight: 140)
                    .focused($editing, equals: .text)
                    .disabled(!editable)
                    .accessibilityLabel("Text to speak")
                    .accessibilityIdentifier("speech.text")
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                Text("Short passages work best. Maximum 255 tokens.")
                    .font(.caption).foregroundStyle(.secondary)
                LabeledContent("Seed") {
                    TextField("Seed", text: Binding(get: { state.seed }, set: { perform(.setSeed($0)) }))
                        .keyboardType(.numberPad)
                        .textFieldStyle(.roundedBorder)
                        .focused($editing, equals: .seed)
                        .disabled(!editable)
                        .accessibilityLabel("Seed")
                        .accessibilityIdentifier("speech.seed")
                }
                if UInt64(state.seed) == nil {
                    Text("Enter a whole-number seed from 0 to 18446744073709551615.")
                        .font(.caption).foregroundStyle(.orange)
                }
                Button("Generate speech", systemImage: "waveform") { perform(.generate) }
                    .buttonStyle(.borderedProminent)
                    .disabled(!state.canGenerate || !isPermitted())
                    .accessibilityIdentifier("speech.generate")
                if state.busy {
                    Button("Stop", systemImage: "stop.fill", role: .destructive) { send(.stop) }
                        .disabled(state.phase == .stopping)
                        .accessibilityIdentifier("speech.stop")
                    ProgressView { Text(LocalizedStringKey(phaseLabel)) }
                } else if let message = state.message {
                    RuntimeMessageText(message).font(.callout).textSelection(.enabled)
                }
                if let audio = state.audio {
                    Divider()
                    HStack {
                        Button("Play", systemImage: "play.fill") { perform(.play) }
                            .accessibilityIdentifier("speech.play")
                        Spacer()
                        Button("Save WAV…", systemImage: "square.and.arrow.up") { perform(.prepareExport) }
                            .accessibilityIdentifier("speech.save")
                    }
                    .disabled(!state.canPlay || !isPermitted())
                    LabeledContent("Audio length", value: seconds(audio.audioSeconds))
                    LabeledContent("Generation", value: seconds(audio.elapsedSeconds))
                    Text("48 kHz · Mono WAV").font(.caption).foregroundStyle(.secondary)
                    DisclosureGroup("Text used for generation") {
                        Text(verbatim: audio.normalizedText).textSelection(.enabled)
                    }
                }
            }
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { editing = nil }
                }
            }
            .background {
                if isPermitted(), let ticket = state.exportTicket, let audio = state.audio {
                    SpeechExportPresenter(ticket: ticket, data: audio.wav, send: perform).id(ticket.id)
                }
            }
        }

        private var editable: Bool { state.editable && isPermitted() }

        private func perform(_ action: SpeechAction) {
            guard isPermitted() else { return }
            switch action {
            case .generate, .play, .prepareExport: editing = nil
            default: break
            }
            send(action)
        }

        private func seconds(_ value: Double) -> String {
            DemoStrings(locale: locale).number(value, fractionDigits: 2...2) + " s"
        }

        private var phaseLabel: String {
            switch state.phase {
            case .idle: "Ready"
            case .opening: "Checking model files…"
            case .generating: "Generating speech…"
            case .playing: "Playing audio…"
            case .stopping: "Waiting for the current operation to stop…"
            }
        }
    }
#endif
