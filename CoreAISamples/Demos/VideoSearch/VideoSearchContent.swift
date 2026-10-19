import AVKit
import SwiftUI
import UniformTypeIdentifiers
import VideoSearchFeature

#if os(iOS)
    import PhotosUI
#endif

struct VideoSearchContent: View {
    let state: VideoSearchState
    let send: @MainActor (VideoSearchAction) -> Void
    let isPermitted: @MainActor () -> Bool
    @State private var player: AVPlayer?
    @State private var playerURL: URL?
    @State private var choosingVideo = false
    @State private var videoRequest = 0
    #if os(iOS)
        @State private var pickedVideo: PhotosPickerItem?
    #endif
    @Environment(\.locale) private var locale

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            controls
            if let failure = state.failure { failureView(failure) }
            if state.needsIndex && !state.isBusy { indexNotice }
            if state.video != nil { playback }
            if state.hasSearched {
                results
            } else if state.video == nil {
                ContentUnavailableView("Find moments in a video with your own words", systemImage: "film.stack")
                    .frame(minHeight: 280)
            }
        }
        #if os(iOS)
            .photosPicker(
                isPresented: $choosingVideo, selection: $pickedVideo, matching: .videos, preferredItemEncoding: .current
            )
            .task(id: pickedVideo) { await importVideo() }
        #else
            .fileImporter(isPresented: $choosingVideo, allowedContentTypes: [.movie]) { result in
                switch result {
                case .success(let url): send(.videoChosen(videoRequest, url))
                case .failure(let error): send(.selectionFailed(videoRequest, error.localizedDescription))
                }
            }
        #endif
        .onChange(of: state, initial: true) { connectPlayer() }
        .onDisappear {
            player?.pause()
            #if os(iOS)
                VideoFile.removeCopies(keeping: [state.video])
            #endif
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            Button("Choose video…") {
                videoRequest = state.session
                choosingVideo = true
            }
            .disabled(!state.canIndex)
            .accessibilityIdentifier("video-search-choose")
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField(
                    "Describe a moment to find", text: Binding(get: { state.query }, set: { send(.setQuery($0)) })
                )
                .textFieldStyle(.plain)
                .onSubmit { send(.search) }
                .accessibilityIdentifier("video-search-query")
                Button("Search") { send(.search) }
                    .buttonStyle(.borderedProminent)
                    .disabled(!state.canSearch)
                    .accessibilityIdentifier("video-search-submit")
            }
            .padding(10)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
            Menu("Example searches") {
                ForEach(VideoSearchAssembly.queries) { query in
                    Button {
                        send(.setQuery(query.text))
                    } label: {
                        Text(verbatim: query.text)
                    }
                }
            }
            .disabled(state.isBusy)
            .accessibilityIdentifier("video-search-examples")
            status
            DisclosureGroup("Search settings") {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("Match threshold")
                        Spacer()
                        Text(state.minimumSimilarity, format: .number.precision(.fractionLength(2)))
                    }
                    Slider(
                        value: Binding(get: { state.minimumSimilarity }, set: { send(.thresholdChanged($0)) }),
                        in: VideoSearchState.thresholdRange, step: 0.01
                    )
                    .accessibilityLabel("Match threshold")
                    Text(
                        "The default comes from the image demo and is not calibrated for video. Adjust it for your video."
                    )
                    .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.top, 8)
                .disabled(!state.canChangeSettings)
            }
        }
    }

    private var status: some View {
        HStack(spacing: 10) {
            if state.isBusy {
                ProgressView().controlSize(.small)
                switch state.phase {
                case .loading: Text("Preparing the model…")
                case .indexing: Text("Indexing frames: \(state.completed) of \(state.total)")
                case .searching: Text("Searching the video…")
                case .stopping: Text("Stopping…")
                case .idle: EmptyView()
                }
                Spacer()
                Button("Stop") { send(.stop) }
                    .disabled(state.phase == .stopping)
            } else {
                Text("Search stays on this device.")
                Spacer()
                if let seconds = state.seconds {
                    let value = DemoStrings(locale: locale).number(seconds, fractionDigits: 2...2)
                    if state.hasSearched { Text("Search: \(value) s") } else { Text("Indexing: \(value) s") }
                }
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private var indexNotice: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Index the video again to search it.", systemImage: "arrow.clockwise")
            Button("Index again") {
                if let video = state.video { send(.index(video)) }
            }
            .buttonStyle(.bordered)
            .disabled(!state.canIndex)
        }
        .font(.callout)
    }

    private var playback: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let player {
                VideoPlayer(player: player)
                    .aspectRatio(16 / 9, contentMode: .fit)
                    .frame(maxHeight: 240)
                    .frame(maxWidth: .infinity)
                    .accessibilityIdentifier("video-search-player")
            } else {
                Text("Playback is available once the model is ready.").font(.footnote).foregroundStyle(.secondary)
            }
            if state.hasSearched && state.duration > 0 { markers }
        }
    }

    private var markers: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary).frame(height: 4)
                ForEach(state.displayedMatches) { match in
                    Button {
                        play(from: match.moment.time)
                    } label: {
                        Capsule().fill(Color.accentColor).frame(width: 4, height: 16)
                            .frame(width: 24, height: 24)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .position(
                        x: geometry.size.width * min(1, max(0, match.moment.time / state.duration)),
                        y: geometry.size.height / 2
                    )
                    .accessibilityLabel(Text("Play from \(timestamp(match.moment.time))"))
                }
            }
        }
        .frame(height: 24)
    }

    private var results: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Search results").font(.title2.bold())
                    Text(verbatim: state.submittedQuery).foregroundStyle(.secondary)
                    if let identity = state.resultIdentity {
                        Text("Results from \(identity)")
                            .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.tail)
                    }
                }
                Spacer()
                Text("\(state.displayedMatches.count) moments").font(.subheadline).foregroundStyle(.secondary)
            }
            Toggle(
                "Show all candidates",
                isOn: Binding(get: { state.showsAllCandidates }, set: { send(.showAllCandidates($0)) })
            )
            .accessibilityIdentifier("video-search-all-candidates")
            Text("Similarity compares meaning; it is not a probability.")
                .font(.caption).foregroundStyle(.secondary)
            if state.displayedMatches.isEmpty {
                ContentUnavailableView(
                    "No close matches", systemImage: "magnifyingglass",
                    description: Text("Try different wording, lower the match threshold, or show all candidates.")
                )
                .accessibilityIdentifier("video-search-no-matches")
            }
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(state.displayedMatches) { match in row(match) }
            }
        }
    }

    private func row(_ match: VideoSearchMatch) -> some View {
        Button {
            play(from: match.moment.time)
        } label: {
            HStack(spacing: 12) {
                if let image = ImageSearchAssembly.image(data: match.moment.thumbnail) {
                    Image(decorative: image, scale: 1).resizable().scaledToFill()
                        .frame(width: 96, height: 54)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                Text(verbatim: timestamp(match.moment.time)).font(.body.monospacedDigit())
                Spacer()
                Text(match.similarity, format: .number.precision(.fractionLength(3)))
                    .monospacedDigit().foregroundStyle(.secondary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text("Play from \(timestamp(match.moment.time))"))
        .accessibilityValue(Text(match.similarity, format: .number.precision(.fractionLength(3))))
    }

    private func failureView(_ failure: VideoSearchFailure) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(LocalizedStringKey(failure.message), systemImage: "exclamationmark.triangle")
            if !failure.detail.isEmpty {
                DisclosureGroup("Details") { Text(verbatim: failure.detail).font(.caption).textSelection(.enabled) }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
    }

    private func timestamp(_ time: Double) -> String {
        Duration.seconds(time).formatted(.time(pattern: .minuteSecond(padMinuteToLength: 2)))
    }

    @discardableResult
    private func connectPlayer() -> AVPlayer? {
        guard let video = state.video, isPermitted() else {
            player?.pause()
            player = nil
            playerURL = nil
            return nil
        }
        if let player, playerURL == video { return player }
        player?.pause()
        let connected = AVPlayer(url: video)
        player = connected
        playerURL = video
        return connected
    }

    private func play(from time: Double) {
        guard isPermitted(), let player = connectPlayer() else { return }
        player.seek(to: CMTime(seconds: time, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
        player.play()
    }

    #if os(iOS)
        private func importVideo() async {
            guard let item = pickedVideo else { return }
            let request = videoRequest
            guard isPermitted() else {
                send(.selectionFailed(request, "The selected model is not available right now. Try again."))
                pickedVideo = nil
                return
            }
            VideoFile.removeCopies(keeping: [state.video])
            do {
                let file = try await item.loadTransferable(type: VideoFile.self)
                guard !Task.isCancelled, pickedVideo == item else {
                    if let file { try? FileManager.default.removeItem(at: file.url) }
                    return
                }
                guard let file else { throw CocoaError(.fileReadNoSuchFile) }
                send(.videoChosen(request, file.url))
                VideoFile.removeCopies(keeping: [state.video, file.url])
            } catch {
                guard !(error is CancellationError), !Task.isCancelled, pickedVideo == item else { return }
                send(.selectionFailed(request, error.localizedDescription))
            }
            pickedVideo = nil
        }
    #endif
}

#if os(iOS)
    struct VideoFile: Transferable {
        static let folder = URL.temporaryDirectory.appending(path: "video-search")
        let url: URL

        static var transferRepresentation: some TransferRepresentation {
            FileRepresentation(importedContentType: .movie) { received in
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                let copy = folder.appending(path: UUID().uuidString).appendingPathExtension(
                    received.file.pathExtension)
                try FileManager.default.copyItem(at: received.file, to: copy)
                return VideoFile(url: copy)
            }
        }

        static func removeCopies(keeping kept: [URL?]) {
            let directory = folder.resolvingSymlinksInPath().pathComponents
            let names = Set(
                kept.compactMap { $0 }
                    .filter { $0.deletingLastPathComponent().resolvingSymlinksInPath().pathComponents == directory }
                    .map(\.lastPathComponent))
            let files =
                (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
            for file in files where !names.contains(file.lastPathComponent) {
                try? FileManager.default.removeItem(at: file)
            }
        }
    }
#endif
