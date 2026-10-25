import FeatureStore
import Foundation
import Testing
import VideoSearchFeature
import VideoSearchRuntime

@Suite("Video search interaction contracts")
struct VideoSearchTests {
    let folder = URL(filePath: "/models/embedding")
    let video = URL(filePath: "/videos/clip.mov")
    let other = URL(filePath: "/videos/other.mov")
    let moment = VideoMoment(time: 1.5, thumbnail: Data())

    func ready() -> VideoSearchState {
        var state = VideoSearchState(modelFolder: folder)
        state.ready = true
        state.video = video
        state.duration = 15
        state.moments = [moment]
        state.indexIdentity = state.currentIndexIdentity
        state.query = "a dog running"
        return state
    }

    func matches(_ pairs: [(similarity: Double, time: Double)]) -> [VideoSearchMatch] {
        pairs.map { VideoSearchMatch(moment: VideoMoment(time: $0.time, thumbnail: Data()), similarity: $0.similarity) }
    }

    func searched() -> VideoSearchState {
        var state = ready()
        _ = VideoSearchFeature.reduce(&state, .search)
        _ = VideoSearchFeature.reduce(&state, .searched(1, matches([(0.9, 10), (0.8, 2), (0.75, 6), (0.5, 13)]), 0.1))
        return state
    }

    func clips(_ pairs: [(Double, Double)]) -> VideoClipPlan {
        VideoClipPlan(ranges: pairs.map { VideoClipRange(start: $0.0, end: $0.1) })
    }

    @Test(
        "VID-01 Sampling places one frame at the center of each one-second interval and spaces long videos evenly up to 300 frames",
        arguments: [(15.0, 15), (1_000.0, 300), (0.4, 1)])
    func sampling(duration: Double, count: Int) {
        let times = VideoSampling.times(duration: duration)
        let step = duration / Double(count)
        #expect(times.count == count)
        for (index, time) in times.enumerated() {
            #expect(abs(time - (Double(index) + 0.5) * step) < 1e-9)
        }
        #expect(times.first.map { $0 > 0 } == true && times.last.map { $0 < duration } == true)
    }

    @Test("VID-01 A fifteen-second video samples the centers 0.5 to 14.5 and a short clip samples its middle")
    func samplingValues() {
        #expect(VideoSampling.times(duration: 15) == (0..<15).map { Double($0) + 0.5 })
        #expect(VideoSampling.times(duration: 0.4) == [0.2])
        let long = VideoSampling.times(duration: 1_000)
        #expect(abs(long[1] - long[0] - 1_000.0 / 300) < 1e-9 && abs(long[299] - (1_000 - 1_000.0 / 600)) < 1e-9)
    }

    @Test(
        "VID-01 A length that is not a whole number of seconds samples the center of each full second and of the remaining part",
        arguments: [(3.5, [0.5, 1.5, 2.5, 3.25]), (1.9, [0.5, 1.45])])
    func samplingPartial(duration: Double, expected: [Double]) {
        let times = VideoSampling.times(duration: duration)
        #expect(times.count == expected.count)
        #expect(zip(times, expected).allSatisfy { abs($0 - $1) < 1e-12 })
    }

    @Test("VID-01 An enormous length samples 300 evenly spaced finite times without overflowing")
    func samplingEnormous() {
        let times = VideoSampling.times(duration: 1e300)
        #expect(times.count == 300 && times.allSatisfy(\.isFinite))
        #expect(zip(times, times.dropFirst()).allSatisfy { $0 < $1 })
        #expect(abs(times[0] - 1e300 / 600) <= 1e288 && times[299] < 1e300)
    }

    @Test(
        "VID-01 A zero, negative or non-finite duration samples nothing",
        arguments: [0, -1, Double.nan, .infinity, -.infinity])
    func samplingRejects(duration: Double) {
        #expect(VideoSampling.times(duration: duration).isEmpty)
    }

    @Test("VID-02 Choosing a video loads the model first, then indexes it, and ignores progress from another session")
    func loadThenIndex() {
        var state = VideoSearchState(modelFolder: folder)
        #expect(VideoSearchFeature.reduce(&state, .search).isEmpty)
        #expect(VideoSearchFeature.reduce(&state, .videoChosen(0, video)) == [.run(.load(1, folder), id: .work)])
        #expect(state.video == video && state.phase == .loading && state.needsIndex)
        #expect(VideoSearchFeature.reduce(&state, .loaded(1)) == [.run(.index(1, video), id: .work)])
        _ = VideoSearchFeature.reduce(&state, .progress(0, 3, 15))
        #expect(state.completed == 0 && state.total == 0)
        _ = VideoSearchFeature.reduce(&state, .progress(1, 3, 15))
        #expect(state.completed == 3 && state.total == 15)
        _ = VideoSearchFeature.reduce(&state, .indexed(0, [moment], 15, 0.5))
        #expect(state.moments.isEmpty && state.isBusy)
        _ = VideoSearchFeature.reduce(&state, .indexed(1, [moment], 15, 0.5))
        #expect(state.hasValidIndex && state.duration == 15 && state.moments == [moment] && state.seconds == 0.5)
        state.query = " \n "
        #expect(!state.canSearch && VideoSearchFeature.reduce(&state, .search).isEmpty)
        var missing = VideoSearchState()
        #expect(VideoSearchFeature.reduce(&missing, .index(video)).isEmpty && missing.video == nil)
    }

    @Test("VID-02 Stop rejects late completions, failures and video choices and cannot start again until it is idle")
    func stop() {
        var state = ready()
        #expect(VideoSearchFeature.reduce(&state, .search) == [.run(.search(1, "a dog running"), id: .work)])
        #expect(VideoSearchFeature.reduce(&state, .search).isEmpty)
        #expect(VideoSearchFeature.reduce(&state, .stop) == [.cancel(.work), .run(.waitForIdle(2), id: .control)])
        #expect(VideoSearchFeature.reduce(&state, .stop).isEmpty)
        _ = VideoSearchFeature.reduce(&state, .searched(1, matches([(1, 1.5)]), 1))
        _ = VideoSearchFeature.reduce(&state, .failed(1, VideoSearchFailure(message: "late")))
        #expect(state.matches.isEmpty && state.failure == nil && !state.hasSearched)
        #expect(!state.canSearch && !state.canIndex && !state.canChangeSettings)
        #expect(VideoSearchFeature.reduce(&state, .search).isEmpty)
        #expect(VideoSearchFeature.reduce(&state, .videoChosen(1, other)).isEmpty)
        #expect(VideoSearchFeature.reduce(&state, .videoChosen(2, other)).isEmpty)
        #expect(state.video == video && state.phase == .stopping)
        _ = VideoSearchFeature.reduce(&state, .stopped(1))
        #expect(state.phase == .stopping)
        _ = VideoSearchFeature.reduce(&state, .stopped(2))
        #expect(state.phase == .idle && state.canSearch && state.hasValidIndex)
        #expect(VideoSearchFeature.reduce(&state, .videoChosen(1, other)).isEmpty && state.video == video)
    }

    @Test(
        "VID-02 Closing keeps the chosen video and results, invalidates only the index, and loading and indexing again restore search"
    )
    func closeAndReload() {
        var state = ready()
        _ = VideoSearchFeature.reduce(&state, .search)
        let results = matches([(0.8, 1.5)])
        _ = VideoSearchFeature.reduce(&state, .searched(1, results, 0.2))
        _ = VideoSearchFeature.reduce(&state, .thresholdChanged(0.5))
        #expect(VideoSearchFeature.reduce(&state, .closed) == [.cancelAll, .run(.close(2), id: .control)])
        #expect(VideoSearchFeature.reduce(&state, .disappeared) == [.cancelAll, .run(.close(3), id: .control)])
        _ = VideoSearchFeature.reduce(&state, .stopped(2))
        #expect(state.isBusy)
        _ = VideoSearchFeature.reduce(&state, .stopped(3))
        #expect(state.video == video && state.moments == [moment] && state.matches == results && state.hasSearched)
        #expect(state.query == "a dog running" && state.minimumSimilarity == 0.5)
        #expect(!state.ready && !state.isBusy && !state.hasValidIndex && state.needsIndex && !state.canSearch)
        #expect(VideoSearchFeature.reduce(&state, .load(folder)) == [.run(.load(4, folder), id: .work)])
        _ = VideoSearchFeature.reduce(&state, .loaded(1))
        #expect(!state.ready)
        _ = VideoSearchFeature.reduce(&state, .loaded(4))
        #expect(state.ready && state.video == video && state.matches == results && !state.hasValidIndex)
        #expect(VideoSearchFeature.reduce(&state, .index(video)) == [.run(.index(5, video), id: .work)])
        _ = VideoSearchFeature.reduce(&state, .indexed(5, [moment], 15, 0.4))
        #expect(state.hasValidIndex && state.canSearch && state.minimumSimilarity == 0.5)
    }

    @Test("VID-02 Opening another model folder discards the video and results and rejects the old completion")
    func switchFolder() {
        var state = ready()
        state.matches = matches([(0.8, 1.5)])
        state.hasSearched = true
        let target = URL(filePath: "/models/embeddinggemma2-int4-int8")
        #expect(VideoSearchFeature.reduce(&state, .load(target)) == [.run(.load(1, target), id: .work)])
        #expect(state.modelFolder == target && !state.ready && state.video == nil && state.moments.isEmpty)
        #expect(state.matches.isEmpty && !state.hasSearched && state.duration == 0)
        #expect(VideoSearchFeature.reduce(&state, .loaded(0)).isEmpty && !state.ready)
        _ = VideoSearchFeature.reduce(&state, .loaded(1))
        #expect(state.ready && state.video == nil && !state.needsIndex)
    }

    @Test(
        "VID-02 Choosing another video discards the previous frames and results and a failed index can be retried")
    func anotherVideo() {
        var state = ready()
        state.matches = matches([(0.8, 1.5)])
        state.hasSearched = true
        #expect(VideoSearchFeature.reduce(&state, .index(other)) == [.run(.index(1, other), id: .work)])
        #expect(state.video == other && state.moments.isEmpty && state.matches.isEmpty && state.duration == 0)
        let failure = VideoSearchFailure(message: "This video could not be read. Choose another video.")
        _ = VideoSearchFeature.reduce(&state, .failed(1, failure))
        #expect(!state.isBusy && state.failure == failure && state.needsIndex && state.canIndex)
        #expect(VideoSearchFeature.reduce(&state, .index(other)) == [.run(.index(2, other), id: .work)])
        #expect(state.failure == nil)
        _ = VideoSearchFeature.reduce(&state, .selectionFailed(1, "stale"))
        #expect(state.failure == nil)
    }

    @Test(
        "VID-03 Neighbor suppression ranks the frames first and folds every frame within three seconds, including exactly three, into the better one"
    )
    func suppression() {
        let ranked = matches([(0.9, 10), (0.85, 11), (0.8, 20), (0.5, 30)])
        #expect(VideoSearchState.suppressNeighbors(ranked, gap: 3).map(\.moment.time) == [10, 20, 30])
        let unordered = matches([(0.7, 13), (0.75, 12.5), (0.9, 10)])
        #expect(VideoSearchState.suppressNeighbors(unordered, gap: 3).map(\.moment.time) == [10])
        let boundary = matches([(0.9, 10), (0.8, 13)])
        #expect(VideoSearchState.suppressNeighbors(boundary, gap: 3).map(\.moment.time) == [10])
        let apart = matches([(0.9, 10), (0.8, 13.5)])
        #expect(VideoSearchState.suppressNeighbors(apart, gap: 3).map(\.moment.time) == [10, 13.5])
        let tied = matches([(0.8, 6), (0.8, 4)])
        #expect(VideoSearchState.suppressNeighbors(tied, gap: 3).map(\.moment.time) == [4])
    }

    @Test(
        "VID-03 Results show matches at or above the threshold by similarity with neighbors folded, and showing all candidates reveals every frame"
    )
    func displayedMatches() {
        var state = ready()
        let results = matches([(0.9, 10), (0.85, 11), (0.8, 20), (0.5, 30)])
        _ = VideoSearchFeature.reduce(&state, .search)
        _ = VideoSearchFeature.reduce(&state, .searched(1, results, 0.1))
        #expect(state.minimumSimilarity == 0.67)
        #expect(state.displayedMatches.map(\.moment.time) == [10, 20])
        _ = VideoSearchFeature.reduce(&state, .showAllCandidates(true))
        #expect(state.displayedMatches == results)
        _ = VideoSearchFeature.reduce(&state, .showAllCandidates(false))
        _ = VideoSearchFeature.reduce(&state, .thresholdChanged(0.8))
        #expect(state.displayedMatches.map(\.moment.time) == [10, 20])
        _ = VideoSearchFeature.reduce(&state, .thresholdChanged(0.81))
        #expect(state.displayedMatches.map(\.moment.time) == [10])
    }

    @Test(
        "VID-03 The threshold changes only while idle and between 0.3 and 0.9, and a rebuilt managed Store keeps it with the video and results"
    )
    func threshold() {
        var state = ready()
        #expect(VideoSearchFeature.reduce(&state, .thresholdChanged(0.5)).isEmpty && state.minimumSimilarity == 0.5)
        for rejected in [0.29, 0.91, -1, .nan, .infinity] {
            _ = VideoSearchFeature.reduce(&state, .thresholdChanged(rejected))
        }
        #expect(state.minimumSimilarity == 0.5)
        _ = VideoSearchFeature.reduce(&state, .thresholdChanged(0.3))
        #expect(state.minimumSimilarity == 0.3)
        _ = VideoSearchFeature.reduce(&state, .thresholdChanged(0.9))
        #expect(state.minimumSimilarity == 0.9)
        _ = VideoSearchFeature.reduce(&state, .search)
        _ = VideoSearchFeature.reduce(&state, .thresholdChanged(0.6))
        _ = VideoSearchFeature.reduce(&state, .stop)
        _ = VideoSearchFeature.reduce(&state, .thresholdChanged(0.6))
        #expect(state.minimumSimilarity == 0.9)
        _ = VideoSearchFeature.reduce(&state, .stopped(2))
        _ = VideoSearchFeature.reduce(&state, .thresholdChanged(0.6))
        #expect(state.minimumSimilarity == 0.6)
        _ = VideoSearchFeature.reduce(&state, .index(video))
        _ = VideoSearchFeature.reduce(&state, .indexed(3, [moment], 15, 0.1))
        #expect(state.minimumSimilarity == 0.6)

        let int4 = URL(filePath: "/models/int4")
        var managed = VideoSearchState(managed: folder, identity: "fp16")
        _ = VideoSearchFeature.reduce(&managed, .thresholdChanged(0.75))
        _ = VideoSearchFeature.reduce(&managed, .index(video))
        _ = VideoSearchFeature.reduce(&managed, .loaded(1))
        _ = VideoSearchFeature.reduce(&managed, .indexed(1, [moment], 15, 0.1))
        _ = VideoSearchFeature.reduce(&managed, .setQuery("a dog"))
        _ = VideoSearchFeature.reduce(&managed, .search)
        _ = VideoSearchFeature.reduce(&managed, .searched(2, matches([(0.8, 1.5)]), 0.1))
        _ = VideoSearchFeature.reduce(&managed, .showAllCandidates(true))
        let rebuilt = VideoSearchState(managed: int4, identity: "int4", retaining: managed)
        #expect(rebuilt.minimumSimilarity == 0.75 && rebuilt.video == video && rebuilt.duration == 15)
        #expect(rebuilt.moments == [moment] && rebuilt.matches == managed.matches && rebuilt.hasSearched)
        #expect(rebuilt.showsAllCandidates && rebuilt.query == "a dog" && rebuilt.submittedQuery == "a dog")
        #expect(rebuilt.resultIdentity == "fp16" && rebuilt.session == managed.session)
        #expect(rebuilt.modelFolder == int4 && rebuilt.resourceIdentity == "int4" && !rebuilt.hasValidIndex)
        #expect(VideoSearchState(managed: int4, identity: "int4").minimumSimilarity == 0.67)
    }

    @Test("VID-02 A managed Store ignores model folder changes and loads its own artifact before indexing")
    func managedLoad() {
        var state = VideoSearchState(managed: folder, identity: "fp16")
        #expect(VideoSearchFeature.reduce(&state, .load(URL(filePath: "/models/int4"))).isEmpty)
        #expect(state.modelFolder == folder && !state.isBusy)
        #expect(VideoSearchFeature.reduce(&state, .index(video)) == [.run(.load(1, folder), id: .work)])
        #expect(VideoSearchFeature.reduce(&state, .loaded(1)) == [.run(.index(1, video), id: .work)])
        #expect(state.ready && state.phase == .indexing)
    }

    @Test(
        "NOW-01 The list and the selected time give the position and the previous and next moments, a time not in the list or an empty list gives no selection, and the first has no previous and the last no next"
    )
    func momentSelection() {
        let ranked = matches([(0.9, 30), (0.8, 10), (0.7, 20)])
        let middle = VideoSearchState.selection(in: ranked, at: 10)
        #expect(middle?.index == 1 && middle?.previous == ranked[0] && middle?.next == ranked[2])
        let first = VideoSearchState.selection(in: ranked, at: 30)
        #expect(first?.index == 0 && first?.previous == nil && first?.next == ranked[1])
        let last = VideoSearchState.selection(in: ranked, at: 20)
        #expect(last?.index == 2 && last?.previous == ranked[1] && last?.next == nil)
        let single = VideoSearchState.selection(in: [ranked[1]], at: 10)
        #expect(single?.index == 0 && single?.previous == nil && single?.next == nil)
        #expect(VideoSearchState.selection(in: ranked, at: 15) == nil)
        #expect(VideoSearchState.selection(in: ranked, at: nil) == nil)
        #expect(VideoSearchState.selection(in: [], at: 10) == nil)
    }

    @Test(
        "CLIP-01 Selected times become clips in time order, overlapping or touching clips join, the last clip ends at the video's end, and the total is the joined length"
    )
    func clipPlan() {
        #expect(VideoClipPlan.make(times: [12, 1, 7], length: 2, duration: 15) == clips([(1, 3), (7, 9), (12, 14)]))
        #expect(VideoClipPlan.make(times: [12, 1, 7], length: 2, duration: 15).totalSeconds == 6)
        let five = VideoClipPlan.make(times: [40, 0, 30, 10, 20], length: 3, duration: 60)
        #expect(five.ranges.count == 5 && five.totalSeconds == 15)
        #expect(VideoClipPlan.make(times: [2, 1], length: 3, duration: 15) == clips([(1, 5)]))
        #expect(VideoClipPlan.make(times: [4, 1], length: 3, duration: 15) == clips([(1, 7)]))
        #expect(VideoClipPlan.make(times: [4.5, 1], length: 3, duration: 15) == clips([(1, 4), (4.5, 7.5)]))
        #expect(VideoClipPlan.make(times: [3, 3], length: 3, duration: 15) == clips([(3, 6)]))
        #expect(VideoClipPlan.make(times: [1, 3, 5], length: 10, duration: 15) == clips([(1, 15)]))
        let trimmed = VideoClipPlan.make(times: [13], length: 5, duration: 15)
        #expect(trimmed == clips([(13, 15)]) && trimmed.totalSeconds == 2)
    }

    @Test(
        "CLIP-01 An empty selection, times outside the video and a clip length of zero or less give no clips",
        arguments: [
            ([Double](), 3.0), ([-1, 15, 20, .nan], 3), ([1, 7], 0), ([1, 7], -3), ([1, 7], .nan),
        ])
    func clipPlanEmpty(times: [Double], length: Double) {
        let plan = VideoClipPlan.make(times: times, length: length, duration: 15)
        #expect(plan.ranges.isEmpty && plan.totalSeconds == 0)
    }

    @Test(
        "CLIP-02 Checking, selecting all, clearing and choosing a clip length work while idle, export starts with the plan fixed, and those changes are rejected while exporting"
    )
    func exportSelection() {
        var state = searched()
        let destination = URL(filePath: "/exports/moments.mp4")
        #expect(state.displayedMatches.map(\.moment.time) == [10, 2, 6] && !state.canExport)
        _ = VideoSearchFeature.reduce(&state, .toggleMoment(10))
        _ = VideoSearchFeature.reduce(&state, .toggleMoment(2))
        _ = VideoSearchFeature.reduce(&state, .toggleMoment(10))
        #expect(state.selectedMoments == [2] && state.exportSelection == [2] && state.canExport)
        _ = VideoSearchFeature.reduce(&state, .selectAllMoments)
        #expect(state.exportSelection == [2, 6, 10])
        _ = VideoSearchFeature.reduce(&state, .clearMomentSelection)
        #expect(state.selectedMoments.isEmpty && !state.canExport)
        #expect(VideoSearchFeature.reduce(&state, .export(destination)).isEmpty && !state.isBusy)
        for rejected in [4, 0, -3, .nan, .infinity] {
            _ = VideoSearchFeature.reduce(&state, .setClipLength(rejected))
        }
        #expect(state.clipLength == 3)
        _ = VideoSearchFeature.reduce(&state, .setClipLength(10))
        #expect(state.clipLength == 10)
        _ = VideoSearchFeature.reduce(&state, .setClipLength(3))
        _ = VideoSearchFeature.reduce(&state, .toggleMoment(2))
        _ = VideoSearchFeature.reduce(&state, .toggleMoment(6))
        let plan = clips([(2, 5), (6, 9)])
        #expect(state.exportSelection == [2, 6] && state.exportPlan == plan && plan.totalSeconds == 6)
        #expect(
            VideoSearchFeature.reduce(&state, .export(destination)) == [
                .run(.export(2, plan, video, destination), id: .work)
            ])
        #expect(state.phase == .exporting && state.exportProgress == 0 && state.exportedVideo == nil)
        #expect(VideoSearchFeature.reduce(&state, .export(destination)).isEmpty)
        _ = VideoSearchFeature.reduce(&state, .selectAllMoments)
        #expect(state.selectedMoments == [2, 6])
        _ = VideoSearchFeature.reduce(&state, .toggleMoment(2))
        _ = VideoSearchFeature.reduce(&state, .toggleMoment(10))
        _ = VideoSearchFeature.reduce(&state, .clearMomentSelection)
        _ = VideoSearchFeature.reduce(&state, .setClipLength(5))
        #expect(state.selectedMoments == [2, 6] && state.clipLength == 3)
        _ = VideoSearchFeature.reduce(&state, .exportProgress(2, 0.4))
        #expect(state.exportProgress == 0.4)
        _ = VideoSearchFeature.reduce(&state, .exported(2, destination, 6))
        #expect(state.phase == .idle && state.exportedVideo == destination && state.exportedSeconds == 6)
        #expect(state.selectedMoments == [2, 6] && state.canExport)
    }

    @Test(
        "CLIP-02 Choosing another video and searching again clear the checked moments, and only another video drops the exported clip"
    )
    func exportSelectionCleared() {
        var state = searched()
        let destination = URL(filePath: "/exports/moments.mp4")
        _ = VideoSearchFeature.reduce(&state, .selectAllMoments)
        _ = VideoSearchFeature.reduce(&state, .export(destination))
        _ = VideoSearchFeature.reduce(&state, .exported(2, destination, 9))
        _ = VideoSearchFeature.reduce(&state, .search)
        #expect(state.selectedMoments.isEmpty && state.exportedVideo == destination)
        _ = VideoSearchFeature.reduce(&state, .searched(3, matches([(0.9, 10), (0.8, 2)]), 0.1))
        _ = VideoSearchFeature.reduce(&state, .selectAllMoments)
        _ = VideoSearchFeature.reduce(&state, .index(video))
        #expect(state.selectedMoments.isEmpty && state.exportedVideo == destination)
        _ = VideoSearchFeature.reduce(&state, .indexed(4, [moment], 15, 0.1))
        _ = VideoSearchFeature.reduce(&state, .index(other))
        #expect(state.selectedMoments.isEmpty && state.exportedVideo == nil && !state.canExport)
    }

    @Test(
        "CLIP-02 A late completion, progress or failure from an old session is ignored, and stop cancels the export and stays stopping until cleanup ends"
    )
    func exportStop() {
        var state = searched()
        let destination = URL(filePath: "/exports/moments.mp4")
        _ = VideoSearchFeature.reduce(&state, .selectAllMoments)
        _ = VideoSearchFeature.reduce(&state, .export(destination))
        #expect(VideoSearchFeature.reduce(&state, .stop) == [.cancel(.work), .run(.waitForIdle(3), id: .control)])
        _ = VideoSearchFeature.reduce(&state, .exportProgress(2, 0.9))
        _ = VideoSearchFeature.reduce(&state, .exported(2, destination, 9))
        _ = VideoSearchFeature.reduce(&state, .failed(2, VideoSearchFailure(message: "late")))
        #expect(state.phase == .stopping && state.exportProgress == 0 && state.exportedVideo == nil)
        #expect(state.failure == nil && !state.canExport)
        _ = VideoSearchFeature.reduce(&state, .toggleMoment(2))
        _ = VideoSearchFeature.reduce(&state, .stopped(2))
        #expect(state.phase == .stopping && state.selectedMoments == [2, 6, 10])
        _ = VideoSearchFeature.reduce(&state, .stopped(3))
        #expect(state.phase == .idle && state.canExport && state.exportedVideo == nil)
        #expect(VideoSearchFeature.reduce(&state, .export(destination)).count == 1 && state.session == 4)
        _ = VideoSearchFeature.reduce(&state, .exported(2, destination, 9))
        #expect(state.phase == .exporting && state.exportedVideo == nil)
        let failure = VideoSearchFailure(message: "The clip could not be exported.")
        _ = VideoSearchFeature.reduce(&state, .failed(4, failure))
        #expect(state.phase == .idle && state.failure == failure && state.exportedVideo == nil && state.canExport)
    }

    @Test("CLIP-06 A save failure from an old session is ignored and one from the current session shows its message")
    func saveFailure() {
        var state = searched()
        let destination = URL(filePath: "/exports/moments.mp4")
        _ = VideoSearchFeature.reduce(&state, .selectAllMoments)
        _ = VideoSearchFeature.reduce(&state, .export(destination))
        _ = VideoSearchFeature.reduce(&state, .exported(2, destination, 9))
        #expect(VideoSearchFeature.reduce(&state, .saveFailed(1, "stale")).isEmpty && state.failure == nil)
        _ = VideoSearchFeature.reduce(&state, .saveFailed(2, "disk full"))
        #expect(state.failure == VideoSearchFailure(message: "The clip could not be saved.", detail: "disk full"))
        #expect(state.exportedVideo == destination && !state.isBusy)
    }

    @Test(
        "CLIP-02 A rebuilt managed Store keeps the checked moments, clip length and exported clip but not the export progress"
    )
    func exportRetained() {
        var managed = VideoSearchState(managed: folder, identity: "fp16")
        managed.selectedMoments = [2, 6]
        managed.clipLength = 5
        managed.exportedVideo = URL(filePath: "/exports/moments.mp4")
        managed.exportedSeconds = 10
        managed.exportProgress = 0.5
        managed.phase = .exporting
        let rebuilt = VideoSearchState(managed: URL(filePath: "/models/int4"), identity: "int4", retaining: managed)
        #expect(rebuilt.selectedMoments == [2, 6] && rebuilt.clipLength == 5)
        #expect(rebuilt.exportedVideo == managed.exportedVideo && rebuilt.exportedSeconds == 10)
        #expect(rebuilt.exportProgress == 0 && rebuilt.phase == .idle)
    }

    @MainActor
    @Test(
        "CLIP-02 The real Store and runner keep stop active until an in-flight export returns and drop its late result",
        .timeLimit(.minutes(1)))
    func storeExportCancellation() async {
        let service = SuspendedVideoService(moment: moment)
        let runner = VideoSearchRunner(service: service)
        let stopped = AsyncStream<Void>.makeStream()
        var state = searched()
        _ = VideoSearchFeature.reduce(&state, .selectAllMoments)
        let store = Store<VideoSearchFeature>(initialState: state) { effect, send in
            await runner.run(effect) { action in
                send(action)
                if case .stopped = action { stopped.continuation.yield(()) }
            }
        }
        store.send(.export(URL(filePath: "/exports/moments.mp4")))
        var entered = service.entered.stream.makeAsyncIterator()
        _ = await entered.next()
        #expect(store.state.phase == .exporting && store.state.exportProgress == 0.5)
        store.send(.stop)
        var waiting = service.waiting.stream.makeAsyncIterator()
        _ = await waiting.next()
        #expect(store.state.phase == .stopping)
        service.release()
        var completed = stopped.stream.makeAsyncIterator()
        _ = await completed.next()
        #expect(store.state.phase == .idle && store.state.exportedVideo == nil && store.state.failure == nil)
        #expect(service.exports == 1 && store.state.canExport)
    }

    @MainActor
    @Test(
        "VID-02 The real Store and runner keep stop active until an in-flight search returns", .timeLimit(.minutes(1)))
    func storeCancellation() async {
        let service = SuspendedVideoService(moment: moment)
        let runner = VideoSearchRunner(service: service)
        let stopped = AsyncStream<Void>.makeStream()
        let store = Store<VideoSearchFeature>(initialState: ready()) { effect, send in
            await runner.run(effect) { action in
                send(action)
                if case .stopped = action { stopped.continuation.yield(()) }
            }
        }
        store.send(.search)
        var entered = service.entered.stream.makeAsyncIterator()
        _ = await entered.next()
        store.send(.stop)
        var waiting = service.waiting.stream.makeAsyncIterator()
        _ = await waiting.next()
        #expect(store.state.phase == .stopping)
        store.send(.search)
        #expect(service.calls == 1)
        service.release()
        var completed = stopped.stream.makeAsyncIterator()
        _ = await completed.next()
        #expect(store.state.phase == .idle && store.state.matches.isEmpty && store.state.canSearch)
    }
}

@MainActor
private final class SuspendedVideoService: VideoSearchService {
    let entered = AsyncStream<Void>.makeStream()
    let waiting = AsyncStream<Void>.makeStream()
    let moment: VideoMoment
    var calls = 0
    var exports = 0
    var continuation: CheckedContinuation<Void, Never>?
    var idle: [CheckedContinuation<Void, Never>] = []
    init(moment: VideoMoment) { self.moment = moment }
    func load(_ folder: URL) async throws {}
    func index(_ video: URL, progress: (Int, Int) -> Void) async throws -> VideoIndex {
        VideoIndex(duration: 15, moments: [moment])
    }
    func search(_ query: String) async throws -> [VideoSearchMatch] {
        calls += 1
        await withCheckedContinuation {
            continuation = $0
            entered.continuation.yield(())
        }
        let waiters = idle
        idle = []
        for waiter in waiters { waiter.resume() }
        return [VideoSearchMatch(moment: moment, similarity: 1)]
    }
    func export(
        _ plan: VideoClipPlan, of video: URL, to destination: URL, progress: (Double) -> Void
    ) async throws -> Double {
        exports += 1
        progress(0.5)
        await withCheckedContinuation {
            continuation = $0
            entered.continuation.yield(())
        }
        let waiters = idle
        idle = []
        for waiter in waiters { waiter.resume() }
        return plan.totalSeconds
    }
    func waitForIdle() async {
        await withCheckedContinuation {
            idle.append($0)
            waiting.continuation.yield(())
        }
    }
    func close() async throws {}
    func release() {
        continuation?.resume()
        continuation = nil
    }
}
