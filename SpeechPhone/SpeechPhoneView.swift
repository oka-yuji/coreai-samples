import FeatureStore
import SpeechFeature
import SwiftUI

struct SpeechPhoneView: View {
    @Bindable var speech: Store<SpeechFeature>
    @Bindable var library: Store<SpeechLibraryFeature>
    @Environment(\.scenePhase) private var scenePhase
    private enum Field: Hashable { case text, seed }
    @FocusState private var editing: Field?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    if let model = speech.state.model {
                        Label(model.name, systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    }
                    ForEach(library.state.folders) { folder in
                        Button(folder.name, systemImage: "folder") {
                            editing = nil
                            speech.send(.openBundle(folder.url.path))
                        }
                        .disabled(!speech.state.editable)
                    }
                    Button("Refresh model folders", systemImage: "arrow.clockwise") { library.send(.refresh) }
                        .disabled(library.state.loading || speech.state.busy)
                    if library.state.loading { ProgressView("Finding models…") }
                    if let message = library.state.message { Text(message).foregroundStyle(.secondary) }
                } header: {
                    Text("Model")
                } footer: {
                    Text(
                        "Copy the prepared model folder to Irodori Speech in the Files app, then refresh. Model files use about 3.38 GB; additional space is needed for preparation."
                    )
                }
                Section("Text") {
                    Menu("Examples") {
                        ForEach(speech.state.examples) { example in
                            Button(example.title) {
                                speech.send(.setText(example.text))
                                editing = nil
                            }
                        }
                    }
                    .disabled(!speech.state.editable)
                    TextEditor(text: Binding(get: { speech.state.text }, set: { speech.send(.setText($0)) }))
                        .frame(minHeight: 140)
                        .focused($editing, equals: .text)
                        .disabled(!speech.state.editable)
                        .accessibilityLabel("Text to speak")
                        .accessibilityIdentifier("speech.text")
                    Text("Short passages work best. Maximum 255 tokens.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Voice variation") {
                    TextField("Seed", text: Binding(get: { speech.state.seed }, set: { speech.send(.setSeed($0)) }))
                        .keyboardType(.numberPad)
                        .focused($editing, equals: .seed)
                        .disabled(!speech.state.editable)
                        .accessibilityLabel("Seed")
                    if UInt64(speech.state.seed) == nil {
                        Text("Enter a whole-number seed from 0 to 18446744073709551615.")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
                Section {
                    Button("Generate speech", systemImage: "waveform") {
                        editing = nil
                        speech.send(.generate)
                    }
                    .disabled(!speech.state.canGenerate)
                    .accessibilityIdentifier("speech.generate")
                    if speech.state.busy {
                        Button("Stop", systemImage: "stop.fill", role: .destructive) { speech.send(.stop) }
                            .disabled(speech.state.phase == .stopping)
                            .accessibilityIdentifier("speech.stop")
                    }
                    status
                }
                if let audio = speech.state.audio {
                    Section("Audio") {
                        Button("Play", systemImage: "play.fill") {
                            editing = nil
                            speech.send(.play)
                        }
                        .disabled(!speech.state.canPlay)
                        .accessibilityIdentifier("speech.play")
                        Button("Save WAV…", systemImage: "square.and.arrow.up") {
                            editing = nil
                            speech.send(.prepareExport)
                        }
                        .disabled(!speech.state.canPlay)
                        .accessibilityIdentifier("speech.save")
                        LabeledContent(
                            "Audio length",
                            value: "\(audio.audioSeconds.formatted(.number.precision(.fractionLength(2)))) s")
                        LabeledContent(
                            "Generation",
                            value: "\(audio.elapsedSeconds.formatted(.number.precision(.fractionLength(2)))) s")
                        Text("48 kHz · Mono WAV").font(.caption).foregroundStyle(.secondary)
                        DisclosureGroup("Text used for generation") {
                            Text(audio.normalizedText).textSelection(.enabled)
                        }
                    }
                }
            }
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle("Irodori Speech")
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { editing = nil }
                }
            }
        }
        .onAppear { appear() }
        .onDisappear { end() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { end() } else if phase == .active { appear() }
        }
        .background {
            if let ticket = speech.state.exportTicket, let audio = speech.state.audio {
                SpeechExportPresenter(ticket: ticket, data: audio.wav, send: speech.send).id(ticket.id)
            }
        }
    }
    @ViewBuilder private var status: some View {
        if speech.state.busy {
            HStack {
                ProgressView()
                Text(phaseLabel)
            }
        } else if let message = speech.state.message {
            Text(message).textSelection(.enabled)
        }
    }
    private var phaseLabel: String {
        switch speech.state.phase {
        case .idle: "Ready"
        case .opening: "Checking model files…"
        case .generating: "Generating speech…"
        case .playing: "Playing audio…"
        case .stopping: "Waiting for the current operation to stop…"
        }
    }
    private func appear() {
        speech.send(.appeared)
        library.send(.appeared)
    }
    private func end() {
        speech.send(.ended)
        library.send(.ended)
    }
}
