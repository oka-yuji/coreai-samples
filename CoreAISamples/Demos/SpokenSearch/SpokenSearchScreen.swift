import FeatureStore
import SpokenSearchFeature
import SwiftUI

struct SpokenSearchScreen: View {
    let store: Store<SpokenSearchFeature>
    @Environment(\.locale) private var locale
    @Environment(\.scenePhase) private var phase
    @State private var picker: (id: UUID, generation: Int)?

    var body: some View {
        let state = store.state
        let generation = state.generation
        let mode = state.mode
        let label: LocalizedStringKey = mode == .word ? "Word to find" : "Question or topic"
        let example: LocalizedStringKey =
            mode == .word ? "Example word from the recording" : "Example question about the recording"
        let canSearch =
            state.hasTranscript && !state.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !state.isWorking && (mode == .word || state.hasTopicModel)
        VStack(alignment: .leading, spacing: 16) {
            Text("Choose a Japanese recording. Apple Speech can download Japanese recognition data the first time.")
                .font(.callout).foregroundStyle(.secondary)
            Button {
                picker = (UUID(), generation)
            } label: {
                Label("Choose a recording…", systemImage: "waveform")
            }
            .buttonStyle(.bordered)
            if let filename = state.filename { Text(verbatim: filename).font(.headline).textSelection(.enabled) }
            if state.isWorking {
                HStack {
                    if state.isStoppingTopic {
                        ProgressView("Stopping…")
                    } else if state.hasTranscript {
                        if mode == .word {
                            ProgressView("Searching the transcript…")
                        } else {
                            ProgressView("Finding related passages…")
                        }
                    } else {
                        ProgressView("Transcribing Japanese speech…")
                    }
                    if !state.isStoppingTopic { Button("Cancel") { store.send(.cancel) } }
                }
            } else if state.hasTranscript && !state.hasSearched {
                Text("Transcript ready").font(.callout).foregroundStyle(.secondary)
            }
            Picker("Search mode", selection: Binding(get: { store.state.mode }, set: { store.send(.modeChanged($0)) }))
            {
                Text("Word").tag(SpokenSearchMode.word)
                Text("Topic").tag(SpokenSearchMode.topic)
            }
            .pickerStyle(.segmented)
            if mode == .topic {
                if state.hasTopicModel {
                    Text(
                        "Ask about the recording. EmbeddingGemma 2 compares your question with the transcribed passages."
                    )
                    .font(.callout).foregroundStyle(.secondary)
                } else {
                    Text(
                        "Download EmbeddingGemma 2 to search by topic. You can search by word while the model is being prepared."
                    )
                    .font(.callout).foregroundStyle(.secondary)
                }
            }
            VStack(alignment: .leading, spacing: 6) {
                Text(label).font(.headline)
                TextField(
                    label, text: Binding(get: { store.state.query }, set: { store.send(.queryChanged($0)) }),
                    prompt: Text(example)
                )
                .textFieldStyle(.roundedBorder)
                .submitLabel(.search)
                .onSubmit { if canSearch { store.send(.search) } }
                .accessibilityIdentifier("spoken-search-query")
                Button("Search") { store.send(.search) }
                    .buttonStyle(.borderedProminent).disabled(!canSearch)
            }
            if let notice = state.notice {
                Text(message(for: notice)).font(.callout).accessibilityIdentifier("spoken-search-notice")
            }
            if state.hasSearched {
                if state.hits.isEmpty {
                    if mode == .word {
                        Text("No spoken matches. Try another word from the recording.")
                    } else {
                        Text("No passages with playback times were found in this transcript.")
                    }
                } else {
                    Text("Search results").font(.headline)
                    if mode == .word {
                        Text("Matches appear in recording order. Play starts at the spoken word.")
                            .font(.callout).foregroundStyle(.secondary)
                    } else {
                        Text(
                            "Most relevant passages appear first. Play starts at the beginning of each passage in the original recording."
                        )
                        .font(.callout).foregroundStyle(.secondary)
                    }
                    LazyVStack(alignment: .leading, spacing: 16) {
                        ForEach(state.hits) { hit in
                            let start = time(hit.start)
                            let end = time(hit.end)
                            let playing = state.playingID == hit.id
                            VStack(alignment: .leading, spacing: 6) {
                                Text(verbatim: hit.text).font(.headline)
                                Text("From \(start) to \(end)").monospacedDigit().font(.callout)
                                HStack {
                                    Button {
                                        guard store.state.generation == generation, store.state.mode == mode,
                                            store.state.hits.contains(hit)
                                        else { return }
                                        store.send(.play(hit.id))
                                    } label: {
                                        if playing {
                                            Label("Play again", systemImage: "arrow.counterclockwise")
                                        } else {
                                            Label("Play", systemImage: "play.fill")
                                        }
                                    }
                                    .accessibilityValue(Text("From \(start) to \(end)"))
                                    if playing {
                                        Button("Stop playback", systemImage: "stop.fill") {
                                            if store.state.generation == generation && store.state.mode == mode
                                                && store.state.playingID == hit.id
                                            {
                                                store.send(.stopPlayback)
                                            }
                                        }
                                    }
                                }
                                .buttonStyle(.bordered)
                            }
                            .accessibilityElement(children: .contain)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .disabled(!state.isActive)
        .background {
            if let request = picker {
                SpokenRecordingPicker { result in
                    guard picker?.id == request.id else { return }
                    picker = nil
                    switch result {
                    case .success(let urls):
                        if let url = urls.first { store.send(.fileSelected(url, generation: request.generation)) }
                    case .failure(let error):
                        if (error as? CocoaError)?.code != .userCancelled {
                            store.send(.fileSelectionFailed(generation: request.generation))
                        }
                    }
                }
                .id(request.id)
            }
        }
        .onAppear {
            if phase != .background { store.send(.appeared) }
        }
        .onDisappear { suspend() }
        .onChange(of: phase) { _, next in
            if next == .background { suspend() }
            if next == .active { store.send(.appeared) }
        }
    }

    private func suspend() {
        picker = nil
        store.send(.disappeared)
    }

    private func time(_ seconds: Double) -> String {
        Duration.seconds(seconds).formatted(
            .time(pattern: .minuteSecond(padMinuteToLength: 1, fractionalSecondsLength: 2)).locale(locale))
    }

    private func message(for notice: SpokenNotice) -> LocalizedStringKey {
        switch notice {
        case .unavailable: "On-device speech recognition is unavailable on this device."
        case .languageUnavailable:
            "Japanese recognition data is unavailable. Check your connection and choose the recording again."
        case .unreadableFile: "The audio file could not be read. Choose another recording."
        case .recognitionFailed: "Speech recognition did not finish. Choose the recording again to retry."
        case .noSpeech: "No timed Japanese speech was found in this recording."
        case .playbackFailed: "The recording could not be played. Try the match again."
        case .topicUnavailable:
            "The selected model is unavailable. Close any other demo using a model, check Models and try again."
        case .topicFailed: "The topic search could not finish. Try searching again."
        case .shortenQuery: "The question is too long for EmbeddingGemma 2. Shorten it and search again."
        }
    }
}
