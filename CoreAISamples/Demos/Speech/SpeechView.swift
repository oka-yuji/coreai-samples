import FeatureStore
import SpeechFeature
import SwiftUI

struct SpeechView: View {
    @Bindable var store: Store<SpeechFeature>

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Irodori").font(.title2.bold())
                    Text("Text to speech").font(.headline)
                    Text(store.state.model?.name ?? "Open an Irodori model folder to begin.")
                        .font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                }
                Spacer()
                if store.state.model == nil, let path = store.state.lastPath {
                    Button("Open last model") { store.send(.openBundle(path)) }
                        .disabled(!store.state.editable)
                }
                SpeechModelButton(store: store)
            }
            HStack {
                Text("Text").font(.headline)
                Spacer()
                Menu("Examples") {
                    ForEach(store.state.examples) { example in
                        Button(example.title) { store.send(.setText(example.text)) }
                    }
                }
                .disabled(!store.state.editable || store.state.examples.isEmpty)
                Button("Clear") { store.send(.setText("")) }
                    .disabled(!store.state.editable || store.state.text.isEmpty)
            }
            TextEditor(text: Binding(get: { store.state.text }, set: { store.send(.setText($0)) }))
                .font(.body)
                .autocorrectionDisabled()
                .frame(minHeight: 150, maxHeight: .infinity)
                .padding(6)
                .background(.background, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                .disabled(!store.state.editable)
                .accessibilityLabel("Text to speak")
                .accessibilityIdentifier("speech.text")
            HStack(spacing: 14) {
                TextField("Seed", text: Binding(get: { store.state.seed }, set: { store.send(.setSeed($0)) }))
                    .textFieldStyle(.roundedBorder).frame(width: 220)
                    .disabled(!store.state.editable)
                    .accessibilityIdentifier("speech.seed")
                    .help("Change the seed to try another voice variation.")
                Text("Short passages work best. The limit is 255 tokens.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            if UInt64(store.state.seed) == nil {
                Text("Enter a whole-number seed from 0 to 18446744073709551615.")
                    .font(.caption).foregroundStyle(.orange)
            }
            HStack(spacing: 12) {
                Button("Generate speech", systemImage: "waveform") { store.send(.generate) }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!store.state.canGenerate)
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
                    Text(audio.normalizedText).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                }
                .font(.callout)
            }
            HStack(spacing: 10) {
                if store.state.busy {
                    ProgressView().controlSize(.small)
                    Text(phaseLabel)
                } else if let message = store.state.message {
                    Text(message).textSelection(.enabled)
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
        .navigationTitle("Irodori")
        .onAppear { store.send(.appeared) }
        .onDisappear { store.send(.ended) }
        .background {
            if let ticket = store.state.exportTicket, let audio = store.state.audio {
                SpeechExportPresenter(ticket: ticket, data: audio.wav, send: store.send).id(ticket.id)
            }
        }
    }

    private var phaseLabel: String {
        switch store.state.phase {
        case .idle: "Ready"
        case .opening: "Checking model files…"
        case .generating: "Generating speech…"
        case .playing: "Playing audio…"
        case .stopping: "Stopping after the current operation…"
        }
    }
}
