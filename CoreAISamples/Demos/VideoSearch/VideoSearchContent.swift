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
    @State private var selectedTime: Double?
    @State private var visibleHeight: CGFloat = .infinity
    #if os(iOS)
        @State private var pickedVideo: PhotosPickerItem?
    #endif
    @Environment(\.locale) private var locale

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            controls
            if let failure = state.failure { failureView(failure) }
            if state.needsIndex && !state.isBusy { indexNotice }
            if state.video != nil {
                LazyVStack(alignment: .leading, spacing: 4, pinnedViews: [.sectionHeaders]) {
                    Section {
                        if state.hasSearched {
                            resultsHeader
                            ForEach(state.displayedMatches) { match in row(match) }
                        }
                    } header: {
                        playback
                    }
                }
            } else {
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
        .onChange(of: state.video) { selectedTime = nil }
        .onChange(of: state.matches) { selectedTime = nil }
        .task(id: state.video) { await connectWhenPermitted() }
        .onAppear { VideoExport.removeStaleFiles(keeping: state.exportedVideo) }
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
                case .exporting: Text("Exporting…")
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
                Color.clear
                    .aspectRatio(16 / 9, contentMode: .fit)
                    .overlay { VideoPlayer(player: player).accessibilityIdentifier("video-search-player") }
                    .frame(maxHeight: maximumPlayerHeight)
                    .frame(maxWidth: .infinity)
            } else {
                Text("Playback is available once the model is ready.").font(.footnote).foregroundStyle(.secondary)
            }
            if state.hasSearched && state.duration > 0 { markers }
            if state.hasSearched && !state.displayedMatches.isEmpty { selectionBar }
            if !state.exportSelection.isEmpty || state.phase == .exporting || state.exportedVideo != nil {
                VideoExportBar(state: state, send: send)
            }
        }
        .padding(.bottom, 8)
        .background(.background)
        .overlay(alignment: .bottom) { Divider() }
        .onGeometryChange(for: CGFloat.self) {
            $0.bounds(of: .scrollView)?.height ?? .infinity
        } action: {
            visibleHeight = $0
        }
    }

    private var maximumPlayerHeight: CGFloat {
        guard visibleHeight.isFinite, visibleHeight > 0 else { return 240 }
        return min(240, visibleHeight * 0.4)
    }

    private var markers: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary).frame(height: 4)
                ForEach(state.displayedMatches) { match in
                    let selected = match.moment.time == selectedTime
                    Capsule().fill(Color.accentColor.opacity(selected ? 1 : 0.35))
                        .frame(width: selected ? 5 : 4, height: selected ? 20 : 14)
                        .position(
                            x: geometry.size.width * min(1, max(0, match.moment.time / state.duration)),
                            y: geometry.size.height / 2
                        )
                        .zIndex(selected ? 1 : 0)
                }
            }
        }
        .frame(height: 24)
        .accessibilityHidden(true)
    }

    private var selectionBar: some View {
        let matches = state.displayedMatches
        let selection = VideoSearchState.selection(in: matches, at: selectedTime)
        let previous = selection == nil ? matches.last : selection?.previous
        let next = selection == nil ? matches.first : selection?.next
        return HStack(spacing: 8) {
            if let selection {
                Label("Selected moment", systemImage: "play.fill")
                Text(verbatim: timestamp(matches[selection.index].moment.time)).bold()
                Text(verbatim: "\(selection.index + 1) / \(matches.count)").foregroundStyle(.secondary)
            } else {
                Label("Select a moment", systemImage: "play")
            }
            Spacer()
            step("Previous moment", systemImage: "chevron.backward", to: previous)
            step("Next moment", systemImage: "chevron.forward", to: next)
        }
        .font(.callout.monospacedDigit())
        .frame(minHeight: 44)
    }

    private func step(_ title: LocalizedStringKey, systemImage: String, to target: VideoSearchMatch?) -> some View {
        Button {
            if let target { play(from: target.moment.time) }
        } label: {
            Image(systemName: systemImage).frame(minWidth: 44, minHeight: 44).contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .disabled(target == nil)
        .accessibilityLabel(title)
    }

    private var resultsHeader: some View {
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
                VStack(alignment: .trailing, spacing: 6) {
                    Text("\(state.displayedMatches.count) moments").font(.subheadline).foregroundStyle(.secondary)
                    Button("Select all") { send(.selectAllMoments) }
                        .buttonStyle(.bordered)
                        .disabled(state.isBusy || state.displayedMatches.isEmpty)
                }
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
        }
        .padding(.top, 16)
        .padding(.bottom, 4)
    }

    private func row(_ match: VideoSearchMatch) -> some View {
        let checked = state.selectedMoments.contains(match.moment.time)
        return HStack(spacing: 0) {
            Button {
                send(.toggleMoment(match.moment.time))
            } label: {
                Image(systemName: checked ? "checkmark.circle.fill" : "circle")
                    .font(.title2)
                    .foregroundStyle(checked ? Color.accentColor : Color.secondary)
                    .frame(width: 44, height: 44)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.borderless)
            .disabled(state.isBusy)
            .accessibilityLabel(Text("Include \(timestamp(match.moment.time)) in the export"))
            .accessibilityAddTraits(checked ? .isSelected : [])
            playButton(match)
        }
    }

    private func playButton(_ match: VideoSearchMatch) -> some View {
        let selected = match.moment.time == selectedTime
        return Button {
            play(from: match.moment.time)
        } label: {
            HStack(spacing: 12) {
                if let image = ImageSearchAssembly.image(data: match.moment.thumbnail) {
                    Image(decorative: image, scale: 1).resizable().scaledToFill()
                        .frame(width: 96, height: 54)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(alignment: .bottomLeading) {
                            if selected {
                                Image(systemName: "play.fill").font(.caption2).foregroundStyle(Color.accentColor)
                                    .padding(5).background(.white, in: Circle()).padding(4)
                            }
                        }
                }
                Text(verbatim: timestamp(match.moment.time)).font(.body.monospacedDigit())
                    .fontWeight(selected ? .bold : nil)
                    .foregroundStyle(selected ? Color.accentColor : Color.primary)
                Spacer()
                Text(match.similarity, format: .number.precision(.fractionLength(3)))
                    .monospacedDigit().foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
            .padding(.horizontal, 6)
            .background(Color.accentColor.opacity(selected ? 0.12 : 0), in: RoundedRectangle(cornerRadius: 10))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text("Play from \(timestamp(match.moment.time))"))
        .accessibilityValue(Text(match.similarity, format: .number.precision(.fractionLength(3))))
        .accessibilityAddTraits(selected ? .isSelected : [])
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

    private func connectWhenPermitted() async {
        guard state.video != nil else { return }
        for _ in 0..<50 where player == nil && !Task.isCancelled {
            if isPermitted() {
                connectPlayer()
                return
            }
            do { try await Task.sleep(for: .milliseconds(200)) } catch { return }
        }
    }

    private func play(from time: Double) {
        guard isPermitted(), let player = connectPlayer() else { return }
        selectedTime = time
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
                    .map(\.lastPathComponent) + [VideoExport.folder.lastPathComponent])
            let files =
                (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
            for file in files where !names.contains(file.lastPathComponent) {
                try? FileManager.default.removeItem(at: file)
            }
        }
    }
#endif

struct VideoExport: Transferable {
    static let folder = URL.temporaryDirectory.appending(path: "video-search/exports")
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .mpeg4Movie) { SentTransferredFile($0.url) }
            .suggestedFileName { $0.url.lastPathComponent }
    }

    static func destination(replacing previous: URL?) -> URL {
        if let previous, isInFolder(previous) { try? FileManager.default.removeItem(at: previous) }
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let suffix = UUID().uuidString.prefix(8).lowercased()
        return folder.appending(path: "moments-\(formatter.string(from: .now))-\(suffix).mp4")
    }

    static func removeStaleFiles(keeping kept: URL?) {
        let key = URLResourceKey.contentModificationDateKey
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [key])) ?? []
        let limit = Date.now.addingTimeInterval(-24 * 60 * 60)
        for file in files where file.lastPathComponent != kept?.lastPathComponent {
            guard let modified = try? file.resourceValues(forKeys: [key]).contentModificationDate, modified < limit
            else { continue }
            try? FileManager.default.removeItem(at: file)
        }
    }

    private static func isInFolder(_ url: URL) -> Bool {
        url.deletingLastPathComponent().resolvingSymlinksInPath().pathComponents
            == folder.resolvingSymlinksInPath().pathComponents
    }
}

private struct VideoExportBar: View {
    let state: VideoSearchState
    let send: @MainActor (VideoSearchAction) -> Void
    @State private var saving = false
    @State private var saveRequest = 0
    @Environment(\.locale) private var locale

    var body: some View {
        let strings = DemoStrings(locale: locale)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) { exportSelectionButtons }
            clipLengthPicker
            if state.phase == .exporting {
                HStack(spacing: 12) {
                    ProgressView(value: state.exportProgress) {
                        Text("Exporting…")
                    } currentValueLabel: {
                        Text(state.exportProgress, format: .percent.precision(.fractionLength(0)))
                    }
                    Button("Stop") { send(.stop) }
                        .disabled(state.phase == .stopping)
                }
            } else {
                Button {
                    send(.export(VideoExport.destination(replacing: state.exportedVideo)))
                } label: {
                    Text(
                        "Export \(state.exportSelection.count) moments (\(strings.number(state.exportPlan.totalSeconds)) s)"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!state.canExport)
            }
            if let video = state.exportedVideo { exportedCard(video, seconds: strings.number(state.exportedSeconds)) }
        }
        .font(.callout)
    }

    private var exportSelectionButtons: some View {
        Group {
            Button("Select all") { send(.selectAllMoments) }
            Button("Clear selection") { send(.clearMomentSelection) }
        }
        .buttonStyle(.bordered)
        .disabled(state.isBusy)
    }

    private var clipLengthPicker: some View {
        HStack(spacing: 8) {
            Text("Clip length").font(.caption).foregroundStyle(.secondary).accessibilityHidden(true)
            Picker("Clip length", selection: Binding(get: { state.clipLength }, set: { send(.setClipLength($0)) })) {
                ForEach(VideoSearchState.clipLengths, id: \.self) { length in Text("\(Int(length)) s").tag(length) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .disabled(state.isBusy)
        }
    }

    private func exportedCard(_ video: URL, seconds: String) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Exported (\(seconds) s)").bold()
                Text(verbatim: video.lastPathComponent)
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
            Spacer(minLength: 0)
            #if os(iOS)
                ShareLink(item: VideoExport(url: video), preview: SharePreview(video.lastPathComponent)) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.borderedProminent)
            #else
                Button("Save…") {
                    saveRequest = state.session
                    saving = true
                }
                .buttonStyle(.borderedProminent)
                .fileExporter(
                    isPresented: $saving, item: VideoExport(url: video), contentTypes: [.mpeg4Movie],
                    defaultFilename: video.lastPathComponent
                ) { result in
                    guard case .failure(let error) = result, (error as? CocoaError)?.code != .userCancelled else {
                        return
                    }
                    send(.saveFailed(saveRequest, error.localizedDescription))
                }
            #endif
        }
        .padding(12)
        .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }
}
