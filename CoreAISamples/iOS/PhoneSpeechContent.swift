#if os(iOS)
    import SpeechFeature
    import SwiftUI

    struct PhoneSpeechContent: View {
        let state: SpeechState
        let send: (SpeechAction) -> Void
        let isPermitted: @MainActor () -> Bool
        @Environment(\.locale) private var locale
        @State private var showingVoiceOptions = false
        private enum Field: Hashable { case text, seed }
        @FocusState private var editing: Field?

        var body: some View {
            VStack(alignment: .leading, spacing: 16) {
                SpeechReferenceControls(state: state, send: perform)
                Divider()
                HStack {
                    Text("3. Enter what to say").font(.headline)
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
                Text("Type a short Japanese passage, or start with an example below.")
                    .font(.callout).foregroundStyle(.secondary)
                if let example = state.examples.first {
                    Button("Try an example", systemImage: "text.badge.plus") {
                        perform(.setText(example.text))
                    }
                    .disabled(!editable)
                    .accessibilityIdentifier("speech.example")
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
                DisclosureGroup("Voice options", isExpanded: $showingVoiceOptions) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(
                            "Leave this number unchanged to get started. Change it to try another variation of the speech."
                        )
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
                    }
                    .padding(.top, 8)
                }
                if UInt64(state.seed) == nil {
                    Text("Enter a whole-number seed from 0 to 18446744073709551615.")
                        .font(.caption).foregroundStyle(.orange)
                }
                Divider()
                Text("4. Generate and listen").font(.headline)
                Button("Generate speech", systemImage: "waveform") { perform(.generate) }
                    .buttonStyle(.borderedProminent)
                    .disabled(!state.canGenerate || !isPermitted())
                    .accessibilityIdentifier("speech.generate")
                if editable && state.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text("Enter text or choose an example to enable generation.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                if state.busy {
                    Button("Stop", systemImage: "stop.fill", role: .destructive) { send(.stop) }
                        .disabled(state.phase == .stopping)
                        .accessibilityIdentifier("speech.stop")
                    ProgressView { Text(LocalizedStringKey(phaseLabel)) }
                    if state.phase == .generating {
                        Text("Keep this screen open. Preparing the model for the first time can take longer.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
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
                } else if !state.busy {
                    Text("When generation finishes, play the audio here or save it as a WAV file.")
                        .font(.callout).foregroundStyle(.secondary)
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
            case .generate, .play, .prepareExport, .recordReference, .finishReferenceRecording, .importReference,
                .playReference:
                editing = nil
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
            case .preparingRecording: "Preparing to record…"
            case .recording: "Recording your reference voice…"
            case .finishingRecording: "Preparing your recorded voice…"
            case .importingReference: "Reading the reference audio…"
            }
        }
    }
#endif
