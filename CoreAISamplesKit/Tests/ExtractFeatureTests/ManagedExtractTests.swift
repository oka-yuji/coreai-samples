import ExtractFeature
import FeatureStore
import Foundation
import GLiNERCore
import Testing

@testable import ExtractRuntime

let boundPath = "/library/artifacts/gliner-fp32/r1"
let boundIdentity = "gliner2.5-multi/gliner-fp32@r1"
let note = "pressure 0.4MPa, noise ok"
let loaded = AssetInfo(path: boundPath, revision: "r1", loadSeconds: 1, floatType: "float32")
let fakeMapping = makeMapping(note, pressure: 9..<15, noise: 23..<25)

func managedState(text: String = note) -> ExtractState {
    var state = ExtractState(form: testForm, labels: testForm)
    state.text = text
    return state.managed(path: boundPath, identity: boundIdentity)
}

func extraction(_ version: Int, _ text: String = note) -> Command<ExtractEffect, ExtractFeature.EffectID> {
    .run(.extract(version: version, text: text, form: testForm, filter: .none), id: .extraction)
}

struct FakeFailure: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

@MainActor
final class FakeExtractService: ExtractServicing {
    enum Call: Equatable {
        case open(URL)
        case extract(String)
        case release
    }

    let started = AsyncStream<Call>.makeStream()
    private(set) var calls: [Call] = []
    var holdsExtraction = false
    var openFailures: [String] = []
    var extractFailures: [String] = []
    var releaseFailures: [String] = []
    private var held: [CheckedContinuation<Void, Never>] = []

    func open(folder: URL) async throws -> AssetInfo {
        record(.open(folder))
        if !openFailures.isEmpty { throw FakeFailure(message: openFailures.removeFirst()) }
        return AssetInfo(path: folder.path(percentEncoded: false), revision: "r1", loadSeconds: 1, floatType: "float32")
    }

    func extract(text: String, schema: FormSchema, filter: CandidateFilter) async throws -> (
        mapping: FormMapping, timing: ExtractTiming
    ) {
        record(.extract(text))
        if holdsExtraction { await withCheckedContinuation { held.append($0) } }
        try Task.checkCancellation()
        if !extractFailures.isEmpty { throw FakeFailure(message: extractFailures.removeFirst()) }
        return (fakeMapping, timing)
    }

    func release() async throws {
        record(.release)
        if !releaseFailures.isEmpty { throw FakeFailure(message: releaseFailures.removeFirst()) }
    }

    func resumeExtractions() {
        let resumed = held
        held = []
        for continuation in resumed { continuation.resume() }
    }

    private func record(_ call: Call) {
        calls.append(call)
        started.continuation.yield(call)
    }
}

@MainActor
final class Permission {
    var granted = true
}

@MainActor
final class ManagedFixture {
    let service: FakeExtractService
    let permission: Permission
    let runner: ManagedExtractRunner
    let store: Store<ExtractFeature>
    let actions: AsyncStream<ExtractAction>

    init(_ state: ExtractState = managedState()) {
        let service = FakeExtractService()
        let permission = Permission()
        let delivered = AsyncStream<ExtractAction>.makeStream()
        let runner = ManagedExtractRunner(
            service: service, isPermitted: { permission.granted }, describe: { $0.localizedDescription })
        self.service = service
        self.permission = permission
        self.runner = runner
        actions = delivered.stream
        store = Store(initialState: state) { effect, send in
            await runner.run(effect) { action in
                send(action)
                delivered.continuation.yield(action)
            }
        }
    }

    var bound: FakeExtractService.Call { .open(URL(filePath: boundPath, directoryHint: .isDirectory)) }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ManagedExtractTests {
    @Test(
        "EXTRACT-01 In the managed mode, appearing, choosing a folder, composing and editing start no load or extraction, and only an explicit request loads the bound artifact and then extracts it"
    )
    func startsOnlyOnRequest() {
        var state = managedState()
        let edited = "pressure 0.5MPa, noise ng"
        let passive = send(
            &state, .appeared, .assetFolderChosen(path: "/other"), .compositionChanged(true),
            .compositionChanged(false), .textChanged(edited))
        #expect(passive == [[], [], [.cancel(.extraction)], [], [.cancel(.extraction)]])
        #expect(state.asset == .none && state.phase == .idle && state.version == 1)

        #expect(send(&state, .extractRequested) == [[.run(.openAsset(path: boundPath), id: .asset)]])
        #expect(state.asset == .opening(path: boundPath) && state.phase == .extracting(version: 1))
        #expect(send(&state, .extractRequested) == [[]])
        #expect(send(&state, .assetOpened(loaded)) == [[extraction(1, edited)]])
        _ = send(&state, .extractionFinished(version: 1, mapping: fakeMapping, timing: timing))
        #expect(state.result == ExtractResult(version: 1, mapping: fakeMapping, timing: timing, assetRevision: "r1"))
        #expect(state.resourceIdentity == boundIdentity && state.phase == .ready)
    }

    @Test(
        "EXTRACT-01 A managed load that ends after an edit replaced its request leaves the model ready without extracting"
    )
    func loadAloneDoesNotExtract() {
        var state = managedState()
        _ = send(&state, .extractRequested, .textChanged("pressure 0.5MPa"))
        #expect(state.phase == .idle)
        #expect(send(&state, .assetOpened(loaded)) == [[]])
        #expect(state.asset == .ready(loaded) && state.result == nil && state.resourceIdentity == nil)
        #expect(send(&state, .extractRequested) == [[extraction(1, "pressure 0.5MPa")]])
    }

    @Test(
        "The standalone lab still opens the remembered folder, extracts when it opens and after a settled edit, and ignores the managed lifecycle actions"
    )
    func standaloneKeepsAutomaticExtraction() {
        var state = ExtractState(form: testForm, labels: testForm, lastAssetPath: "/last")
        state.text = note
        let info = AssetInfo(path: "/last", revision: nil, loadSeconds: 1, floatType: "float32")
        let automatic = send(&state, .appeared, .assetOpened(info), .textChanged("pressure 0.5MPa"))
        #expect(
            automatic == [
                [.run(.openAsset(path: "/last"), id: .asset)], [extraction(0)], [.run(.settle, id: .extraction)],
            ])
        #expect(ExtractFeature.reduce(&state, .compositionChanged(true)) == [.cancel(.extraction)])
        #expect(ExtractFeature.reduce(&state, .compositionChanged(false)) == [.run(.settle, id: .extraction)])
        let before = state
        #expect(send(&state, .closed, .released, .releaseFailed("busy")) == [[], [], []])
        #expect(state == before && state.execution == .standalone)
    }

    @Test(
        "EXTRACT-01 A managed Store calls no service until the explicit request, then loads the bound folder once and extracts the draft"
    )
    func storeLoadsOnRequest() async {
        let fixture = ManagedFixture()
        var actions = fixture.actions.makeAsyncIterator()
        let edited = "pressure 0.5MPa, noise ng"
        fixture.store.send(.appeared)
        fixture.store.send(.assetFolderChosen(path: "/other"))
        fixture.store.send(.compositionChanged(false))
        fixture.store.send(.textChanged(edited))
        fixture.store.send(.extractRequested)
        #expect(await actions.next() == .assetOpened(loaded))
        #expect(await actions.next() == .extractionFinished(version: 1, mapping: fakeMapping, timing: timing))
        #expect(fixture.service.calls == [fixture.bound, .extract(edited)])
        #expect(fixture.store.state.phase == .ready && fixture.store.state.resourceIdentity == boundIdentity)

        fixture.store.send(.extractRequested)
        #expect(await actions.next() == .extractionFinished(version: 1, mapping: fakeMapping, timing: timing))
        #expect(fixture.service.calls == [fixture.bound, .extract(edited), .extract(edited)])
    }

    @Test(
        "EXTRACT-02 Without the gate's permission the runner calls no service for a load or an extraction and sends their failures, while release is not gated"
    )
    func refusesWithoutPermission() async {
        let service = FakeExtractService()
        let runner = ManagedExtractRunner(
            service: service, isPermitted: { false }, describe: { $0.localizedDescription })
        var actions: [ExtractAction] = []
        await runner.run(.openAsset(path: boundPath)) { actions.append($0) }
        await runner.run(.extract(version: 4, text: note, form: testForm, filter: .none)) { actions.append($0) }
        await runner.run(.settle) { actions.append($0) }
        await runner.run(.export(ExportDocument(managedState()))) { actions.append($0) }
        #expect(service.calls.isEmpty)
        #expect(
            actions == [
                .assetFailed(path: boundPath, message: ManagedExtractRunner.unavailable),
                .extractionFailed(version: 4, message: ManagedExtractRunner.unavailable),
            ])
        await runner.run(.release) { actions.append($0) }
        #expect(service.calls == [.release] && actions.last == .released)
    }

    @Test(
        "EXTRACT-02 While the gate is closed for a missing download, a stop or an old session, a direct request reaches the Store as a failure and calls no service"
    )
    func storeRefusesClosedGate() async {
        let fixture = ManagedFixture()
        var actions = fixture.actions.makeAsyncIterator()
        fixture.permission.granted = false
        fixture.store.send(.extractRequested)
        #expect(await actions.next() == .assetFailed(path: boundPath, message: ManagedExtractRunner.unavailable))
        #expect(fixture.store.state.phase == .failed(ManagedExtractRunner.unavailable))
        #expect(fixture.service.calls.isEmpty)

        fixture.permission.granted = true
        fixture.store.send(.extractRequested)
        #expect(await actions.next() == .assetOpened(loaded))
        #expect(await actions.next() == .extractionFinished(version: 0, mapping: fakeMapping, timing: timing))
        fixture.permission.granted = false
        fixture.store.send(.extractRequested)
        #expect(await actions.next() == .extractionFailed(version: 0, message: ManagedExtractRunner.unavailable))
        #expect(fixture.service.calls == [fixture.bound, .extract(note)])
    }

    @Test(
        "EXTRACT-02 Release waits until a cancelled extraction returns, sends nothing for it, and finishes even when its caller is cancelled"
    )
    func releaseJoinsCancelledWork() async throws {
        let service = FakeExtractService()
        service.holdsExtraction = true
        let runner = ManagedExtractRunner(
            service: service, isPermitted: { true }, describe: { $0.localizedDescription })
        var started = service.started.stream.makeAsyncIterator()
        let running = Task {
            var actions: [ExtractAction] = []
            await runner.run(.extract(version: 1, text: note, form: testForm, filter: .none)) { actions.append($0) }
            return actions
        }
        #expect(await started.next() == .extract(note))
        running.cancel()
        let release = Task { try await runner.release() }
        release.cancel()
        await Task.yield()
        await Task.yield()
        #expect(service.calls == [.extract(note)])

        service.resumeExtractions()
        #expect(await running.value.isEmpty)
        try await release.value
        #expect(service.calls == [.extract(note), .release])
    }

    @Test(
        "EXTRACT-02 Closing a managed Store during an extraction cancels it and releases the model only after it returns, without its result"
    )
    func closeDuringExtraction() async throws {
        let fixture = ManagedFixture()
        fixture.service.holdsExtraction = true
        var actions = fixture.actions.makeAsyncIterator()
        var started = fixture.service.started.stream.makeAsyncIterator()
        fixture.store.send(.extractRequested)
        #expect(await started.next() == fixture.bound)
        #expect(await started.next() == .extract(note))
        #expect(await actions.next() == .assetOpened(loaded))

        fixture.store.send(.closed)
        #expect(fixture.store.state.phase == .idle)
        let stop = Task { try await fixture.runner.release() }
        await Task.yield()
        await Task.yield()
        #expect(!fixture.service.calls.contains(.release))

        fixture.service.resumeExtractions()
        try await stop.value
        #expect(await actions.next() == .released)
        #expect(fixture.store.state.asset == .none && fixture.store.state.result == nil)
        #expect(fixture.service.calls == [fixture.bound, .extract(note), .release, .release])
    }

    @Test(
        "EXTRACT-02 A failed load and a failed extraction reach the managed Store, and the next explicit request retries each"
    )
    func failuresAndRetries() async {
        let fixture = ManagedFixture()
        var actions = fixture.actions.makeAsyncIterator()
        fixture.service.openFailures = ["No tokenizer"]
        fixture.store.send(.extractRequested)
        #expect(await actions.next() == .assetFailed(path: boundPath, message: "No tokenizer"))
        #expect(fixture.store.state.phase == .failed("No tokenizer"))
        #expect(fixture.store.state.asset == .failed(path: boundPath, message: "No tokenizer"))

        fixture.service.extractFailures = ["The note is too long"]
        fixture.store.send(.extractRequested)
        #expect(await actions.next() == .assetOpened(loaded))
        #expect(await actions.next() == .extractionFailed(version: 0, message: "The note is too long"))
        #expect(fixture.store.state.phase == .failed("The note is too long"))

        fixture.store.send(.extractRequested)
        #expect(await actions.next() == .extractionFinished(version: 0, mapping: fakeMapping, timing: timing))
        #expect(fixture.store.state.phase == .ready)
        #expect(fixture.service.calls == [fixture.bound, fixture.bound, .extract(note), .extract(note)])
    }

    @Test("EXTRACT-02 Closing a managed state cancels every effect and starts a release, and only a release unloads it")
    func closingReleases() {
        var state = managedState()
        _ = send(&state, .extractRequested, .assetOpened(loaded))
        #expect(state.phase == .extracting(version: 0))
        #expect(send(&state, .closed) == [[.cancelAll, .run(.release, id: .asset)]])
        #expect(state.phase == .idle && state.asset == .ready(loaded))
        _ = send(&state, .releaseFailed("busy"))
        #expect(state.asset == .failed(path: boundPath, message: "busy"))
        #expect(send(&state, .closed, .released) == [[.cancelAll, .run(.release, id: .asset)], []])
        #expect(state.asset == .none)
    }

    @Test(
        "EXTRACT-02 A failed release is thrown to the caller and reported to the Store, and a retried close releases the model"
    )
    func releaseFailureAndRetry() async throws {
        let fixture = ManagedFixture()
        var actions = fixture.actions.makeAsyncIterator()
        fixture.service.releaseFailures = ["The model is busy"]
        fixture.store.send(.closed)
        #expect(await actions.next() == .releaseFailed("The model is busy"))
        #expect(fixture.store.state.asset == .failed(path: boundPath, message: "The model is busy"))
        fixture.store.send(.closed)
        #expect(await actions.next() == .released)
        #expect(fixture.store.state.asset == .none)

        fixture.service.releaseFailures = ["Still busy"]
        await #expect(throws: FakeFailure.self) { try await fixture.runner.release() }
        try await fixture.runner.release()
        #expect(fixture.service.calls == [.release, .release, .release, .release])
    }

    @Test(
        "EXTRACT-03 Stopping and binding a new session keep the unrun draft, the shown result with its artifact and the schema, and the next request loads the new artifact"
    )
    func keepsDraftAndResult() {
        var state = managedState()
        _ = send(
            &state, .extractRequested, .assetOpened(loaded),
            .extractionFinished(version: 0, mapping: fakeMapping, timing: timing), .closed, .released)
        let path = "/library/artifacts/gliner-fp16/r2"
        let identity = "gliner2.5-multi/gliner-fp16@r2"
        var resumed = state.managed(path: path, identity: identity)
        #expect(resumed.text == note && resumed.result == state.result && resumed.resourceIdentity == boundIdentity)
        #expect(resumed.labels == testForm && resumed.form == testForm && resumed.mode == .labels)
        #expect(resumed.asset == .none && resumed.phase == .ready)
        #expect(resumed.execution == .managed(path: path, identity: identity))

        let draft = "pressure 0.6MPa"
        _ = send(&resumed, .textChanged(draft), .extractRequested, .closed)
        var again = resumed.managed(path: path, identity: identity)
        #expect(again.text == draft && again.version == 1 && again.result == nil && again.resourceIdentity == nil)
        #expect(again.asset == .none && again.phase == .idle && again.labels == testForm)
        #expect(send(&again, .extractRequested) == [[.run(.openAsset(path: path), id: .asset)]])
    }
}
