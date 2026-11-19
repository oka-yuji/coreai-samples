import FeatureStore
import SpeechFeature
import SwiftUI

struct SpeechForm: View {
    @Bindable var store: Store<SpeechFeature>
    let title: String
    let limit: LocalizedStringKey
    let missing: LocalizedStringKey?
    let voiceHint: LocalizedStringKey?
    let canGenerate: Bool

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text(verbatim: title).font(.title2.bold())
                        Text("Text to speech").font(.headline)
                        if let opened = store.state.model {
                            Text(opened.name)
                                .font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                        } else if let missing {
                            Text(missing)
                                .font(.callout).foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                }
                HStack {
                    Text("Text").font(.headline)
                    Spacer()
                    Menu("Examples") {
                        ForEach(store.state.examples) { example in
                            Button(LocalizedStringKey("Speech example: " + example.title)) {
                                store.send(.selectExample(example))
                            }
                        }
                    }
                    .disabled(!store.state.editable || store.state.examples.isEmpty)
                    Button("Clear") { store.send(.setText("")) }
                        .disabled(!store.state.editable || store.state.text.isEmpty)
                }
                TextEditor(text: Binding(get: { store.state.text }, set: { store.send(.setText($0)) }))
                    .font(.body)
                    .autocorrectionDisabled()
                    .frame(height: 120)
                    .padding(6)
                    .background(.background, in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                    .disabled(!store.state.editable)
                    .accessibilityLabel("Text to speak")
                    .accessibilityIdentifier("speech.text")
                Text(limit).font(.caption).foregroundStyle(.secondary)
                if let voiceHint {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Voice description").font(.headline)
                        TextField(
                            "Voice description",
                            text: Binding(get: { store.state.voice }, set: { store.send(.setVoice($0)) })
                        )
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                        .disabled(!store.state.editable)
                        .accessibilityIdentifier("speech.voice")
                        Text(voiceHint)
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("Seed").font(.headline)
                    TextField("Seed", text: Binding(get: { store.state.seed }, set: { store.send(.setSeed($0)) }))
                        .textFieldStyle(.roundedBorder).frame(width: 220)
                        .disabled(!store.state.editable)
                        .accessibilityIdentifier("speech.seed")
                    Text(
                        "The seed sets the random starting point of generation. The same input and seed produce the same audio. Change the seed to try a different take, with a different tone, pacing or voice."
                    )
                    .font(.caption).foregroundStyle(.secondary)
                    if UInt64(store.state.seed) == nil {
                        Text("Enter a whole-number seed from 0 to 18446744073709551615.")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
                HStack(spacing: 12) {
                    Button("Generate speech", systemImage: "waveform") { store.send(.generate) }
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(!canGenerate)
                        .accessibilityIdentifier("speech.generate")
                    Button("Play", systemImage: "play.fill") { store.send(.play) }
                        .disabled(!store.state.canPlay)
                        .accessibilityIdentifier("speech.play")
                    if store.state.busy {
                        Button("Stop", systemImage: "stop.fill") { store.send(.stop) }
                            .disabled(store.state.phase == .stopping)
                            .keyboardShortcut(.cancelAction)
                            .accessibilityIdentifier("speech.stop")
                    }
                    Spacer()
                    Button("Save WAV…", systemImage: "square.and.arrow.up") { store.send(.prepareExport) }
                        .disabled(!store.state.canPlay)
                        .accessibilityIdentifier("speech.save")
                }
                Divider()
                if let audio = store.state.audio {
                    HStack {
                        Label(
                            "\(audio.audioSeconds, format: .number.precision(.fractionLength(2))) s audio",
                            systemImage: "speaker.wave.2")
                        Text("Generated in \(audio.elapsedSeconds, format: .number.precision(.fractionLength(2))) s")
                        Spacer()
                        Text("48 kHz · Mono WAV")
                    }
                    .font(.callout).foregroundStyle(.secondary)
                    DisclosureGroup("Text used for generation") {
                        Text(audio.normalizedText).textSelection(.enabled).frame(
                            maxWidth: .infinity, alignment: .leading)
                    }
                    .font(.callout)
                }
                HStack(spacing: 10) {
                    if store.state.busy {
                        ProgressView().controlSize(.small)
                        Text(phaseLabel)
                    } else if let message = store.state.message {
                        RuntimeMessageText(message).textSelection(.enabled)
                    } else {
                        Text("Generate a short passage, then play or save the result.")
                    }
                    Spacer()
                }
                .font(.callout).foregroundStyle(.secondary)
                .accessibilityIdentifier("speech.status")
                .frame(minHeight: 24)
            }
            .padding(20)
        }
    }

    private var phaseLabel: String {
        switch store.state.phase {
        case .idle: "Ready"
        case .opening: "Checking model files…"
        case .generating: "Generating speech…"
        case .playing: "Playing audio…"
        case .stopping: "Stopping after the current operation…"
        case .preparingRecording: "Preparing to record…"
        case .recording: "Recording your reference voice…"
        case .finishingRecording: "Preparing your recorded voice…"
        case .importingReference: "Reading the reference audio…"
        }
    }
}
