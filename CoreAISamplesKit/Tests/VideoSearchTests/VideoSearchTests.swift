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
