import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import SpeechRuntime
import SwiftUI
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct SpeechModelSwitchTests {
    @Test("IOS-IR-04 Models retains the speech lease until generation and audio shutdown join before switching")
    func switchAfterStop() async throws {
        let harness = try RuntimeHarness()
        let speech = SwitchingSpeechAdapter()
        defer {
            speech.service.finish()
            harness.close()
        }
        harness.registry.register(speech)
        let scene = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        var audioEvents = speech.service.events.makeAsyncIterator()
        try await scene.download("text-fp16", &events)
        try await scene.download("image-fp16", &events)
        scene.library.send(.openDemo(.comparison))
        let initial = try await scene.show()
        let store = try #require(speech.store)
        #expect(speech.service.calls.isEmpty)
        store.send(.setText("Test"))
        store.send(.generate)
        try await next(&audioEvents, "generating")
        scene.library.send(.openDemo(.imageSearch))
        try await next(&audioEvents, "stopping")
        #expect(scene.library.state.session == initial)
        #expect(scene.library.state.pendingDestination == .demo(.imageSearch))
        #expect(await harness.leases(initial) == 1)
        #expect(speech.gate?.isPermitted() == false)
        #expect(!store.state.presented && store.state.phase == .stopping)
        speech.service.finish()
        try await next(&events, "A stopDemo")
        #expect(await harness.leases(initial) == 0)
        #expect(store.state.audio == nil && store.state.exportTicket == nil)
        #expect(scene.library.state.session?.selection == .imageSearch)
        _ = try await scene.show()
        #expect(try scene.adapter("image-search").start(.imageSearch, input: "Next model"))
        scene.library.send(.closeDemo)
        try await next(&events, "A stopDemo")
    }
}

@MainActor
private final class SwitchingSpeechAdapter: ModelDemoAdapter {
    let demoID = "comparison"
    let service = SwitchingSpeechService()
    lazy var sessions = SpeechDemoSessions { [service] in service }
    var store: Store<SpeechFeature>?
    var gate: DemoExecutionGate?

    func preview(for selection: DemoSelection) -> AnyView { AnyView(EmptyView()) }
    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        self.gate = gate
        store = sessions.bind(session, examples: [], permits: gate.isPermitted)
        return AnyView(EmptyView())
    }
    func stop(_ session: ModelDemoSession) async throws { await sessions.stop(session) }
}

@MainActor
private final class SwitchingSpeechService: SpeechService {
    let events: AsyncStream<String>
    private let sink: AsyncStream<String>.Continuation
    private var generation: CheckedContinuation<Void, Never>?
    private var stops: [CheckedContinuation<Void, Never>] = []
    private var finished = false
    var calls: [String] = []

    init() { (events, sink) = AsyncStream.makeStream() }
    func open(path: String) -> SpeechModel {
        calls.append("open")
        return SpeechModel(name: "Fixture", directory: URL(filePath: path))
    }
    func generate(text: String, seed: UInt64) async -> SpeechAudio {
        calls.append("generate")
        sink.yield("generating")
        if !finished { await withCheckedContinuation { generation = $0 } }
        return SpeechAudio(wav: Data([1]), audioSeconds: 1, elapsedSeconds: 1, normalizedText: text)
    }
    func play(_ data: Data) {}
    func stop() async {
        calls.append("stop")
        sink.yield("stopping")
        if !finished { await withCheckedContinuation { stops.append($0) } }
    }
    func finish() {
        finished = true
        generation?.resume()
        generation = nil
        let waiting = stops
        stops.removeAll()
        for continuation in waiting { continuation.resume() }
    }
}
