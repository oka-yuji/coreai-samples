import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import SpeechCore
import SpeechFeature

@testable import ModelLibraryRuntime
@testable import SpeechRuntime

@MainActor
enum VoxManagedLeaseScenarios {
    private static func join(_ engine: VoxCPMLeasedEngine, signals: VoxLeaseSignals) async {
        await engine.release()
        signals.record("joined")
    }
    static func engine(
        _ device: VoxLeasePhysicalSpeech, at window: VoxLeaseWindow,
        in fixture: VoxLeaseLibraryFixture
    ) throws -> VoxCPMLeasedEngine {
        let acquirer = fixture.runner.executionLeaseAcquirer(
            for: fixture.model.id, in: try window.current(), runtimeUses: fixture.runtimeUses)
        let signals = device.signals
        return VoxCPMLeasedEngine(engine: device, acquire: acquirer, didRequestRelease: { signals.record("closing") })
    }

    static func closing(manifests: URL, duringOpen: Bool) async throws {
        let f = try VoxLeaseLibraryFixture(manifests: manifests, both: true)
        defer { f.cleanup() }
        let id = VoxLeaseLibraryFixture.new
        let installed = try await f.install(id)
        let staged = try await f.staged(id)
        let w = VoxLeaseWindow(fixture: f)
        let used = try await w.use(id)
        let device = VoxLeasePhysicalSpeech(open: duringOpen, generate: !duringOpen, release: true)
        defer { device.unblock() }
        let e = try engine(device, at: w, in: f)
        let speech = VoxLeaseSpeechService(e).makeStore(device.signals)
        speech.send(.appeared)
        speech.send(.openBundle(used.directory.path))
        if duringOpen {
            try await device.signals.wait("open-enter")
        } else {
            try await device.signals.wait("speech-open-ack")
            try requireLease(speech.state.model?.supportsReferenceAudio == true, "view must have opened")
            speech.send(.setText("probe"))
            speech.send(.generate)
            try await device.signals.wait("generate-enter")
        }
        try await f.counts(installed, physical: 2, displayed: 1)
        speech.send(.ended)
        w.end()
        try await w.events.wait("usage-end")
        f.runner.abandon(owner: w.owner)
        weak let lost = w.library
        w.drop()
        try requireLease(lost == nil, "the product acquirer or release closure retained the library store")
        try await device.signals.wait("closing")
        async let another: Void = join(e, signals: device.signals)
        defer { device.unblock() }
        try await device.signals.wait("closing", 2)
        try await f.counts(installed, physical: 1, displayed: 1)
        try await f.refuseDelete(installed.identity)
        try await f.refuseReplacement(staged)
        try requireLease(!device.signals.log.contains("release-leave"), "model destroyed early")
        device.opening.open()
        device.generating.open()
        try await device.signals.wait("release-enter")
        try await f.counts(installed, physical: 1, displayed: 1)
        try await f.refuseDelete(installed.identity)
        try requireLease(
            !device.signals.log.contains("resource-destroyed"), "test resource was not held at physical stop")
        try requireLease(
            !device.signals.log.contains("joined") && !device.signals.log.contains("speech-stop-ack"),
            "release waiter returned before physical shutdown")
        device.releasing.open()
        await another
        try await device.signals.wait("speech-stop-ack")
        try await f.counts(installed, physical: 0, displayed: 0)
        let log = device.signals.log
        guard let destroyed = log.firstIndex(of: "resource-destroyed"),
            let stopped = log.firstIndex(of: "release-leave"),
            let joined = log.firstIndex(of: "joined"), let acknowledged = log.firstIndex(of: "speech-stop-ack")
        else {
            throw VoxLeaseProbeError(description: "model destruction or stop missing")
        }
        try requireLease(
            destroyed < stopped && stopped < joined && stopped < acknowledged
                && log.count(where: { $0 == "resource-destroyed" }) == 1, "fake model destruction order")
        try requireLease(
            speech.state.audio == nil && !speech.state.presented, "the old completion reached the departed screen")
        if duringOpen { try requireLease(speech.state.model == nil, "old open published a model after view exit") }
        let size = try await f.storage.delete(identity: installed.identity)
        try requireLease(
            size >= installed.sizeBytes && !FileManager.default.fileExists(atPath: installed.directory.path),
            "delete after stop")
    }

    static func replacement(manifests: URL) async throws {
        let f = try VoxLeaseLibraryFixture(manifests: manifests, both: true)
        defer { f.cleanup() }
        let id = VoxLeaseLibraryFixture.new
        let old = try await f.install(id)
        let w = VoxLeaseWindow(fixture: f)
        let a = try await w.use(id)
        let device = VoxLeasePhysicalSpeech()
        let e = try engine(device, at: w, in: f)
        _ = try await e.open(a.directory)
        await e.release()
        w.end()
        try await w.events.wait("usage-end")
        _ = try await f.storage.delete(identity: old.identity)
        let registered = try await f.install(id)
        let b = try await w.use(id)
        try requireLease(
            a == b && a.registrationGeneration != nil && b.registrationGeneration == registered.registrationGeneration
                && a.registrationGeneration != b.registrationGeneration,
            "same identity and path need a new registration")
        do {
            _ = try await e.generate(text: "stale", seed: 1708)
            throw VoxLeaseProbeError(description: "cold stale verification reached generation")
        } catch let failure as VoxLeaseProbeError { throw failure } catch {}
        try requireLease(!device.signals.log.contains("generate-enter"), "stale work reached the inner engine")
        try await f.counts(b, physical: 1, displayed: 0)
        _ = try await e.open(b.directory)
        let ref = try SpeechReferenceAudio(
            samples: Array(repeating: 0.2, count: 80_000), source: .file("synthetic.wav"), format: .voxcpm2)
        let p: any SpeechGenerating = e
        _ = try await p.generate(text: "fresh", seed: 1708, reference: ref)
        try requireLease(
            device.signals.log.contains("reference-80000-fresh-1708"), "fresh registration lost reference witness")
        await e.release()
        w.end()
        try await w.events.wait("usage-end", 2)
        try await f.counts(b, physical: 0, displayed: 0)
    }

    static func windows(manifests: URL, same: Bool) async throws {
        let f = try VoxLeaseLibraryFixture(manifests: manifests, both: true)
        defer { f.cleanup() }
        let id = VoxLeaseLibraryFixture.new
        _ = try await f.install(id)
        _ = try await f.install(VoxLeaseLibraryFixture.old)
        let left = VoxLeaseWindow(fixture: f)
        let right = VoxLeaseWindow(fixture: f)
        let a = try await left.use(id)
        let b = try await right.use(same ? id : VoxLeaseLibraryFixture.old)
        let first = try engine(VoxLeasePhysicalSpeech(), at: left, in: f)
        let second = try engine(VoxLeasePhysicalSpeech(), at: right, in: f)
        _ = try await first.open(a.directory)
        _ = try await second.open(b.directory)
        try await f.counts(a, physical: same ? 4 : 2, displayed: 2)
        await f.runner.release(owner: left.owner)
        try await f.counts(a, physical: same ? 3 : 1, displayed: 2)
        try await f.refuseDelete(a.identity)
        await first.release()
        try await f.counts(b, physical: 2, displayed: 1)
        _ = try await second.generate(text: "still here", seed: 42)
        await second.release()
        right.end()
        try await right.events.wait("usage-end")
        try await f.counts(a, physical: 0, displayed: 0)
        try await f.counts(b, physical: 0, displayed: 0)
    }

    static func lostWindow(manifests: URL) async throws {
        let f = try VoxLeaseLibraryFixture(manifests: manifests, both: false)
        defer { f.cleanup() }
        let r = try await f.install(VoxLeaseLibraryFixture.old)
        let w = VoxLeaseWindow(fixture: f)
        _ = try await w.use(VoxLeaseLibraryFixture.old)
        let get = f.runner.executionLeaseAcquirer(for: f.model.id, in: try w.current(), runtimeUses: f.runtimeUses)
        let token = try await get(r.directory, r.registrationGeneration)
        let substitute = Store<ModelLibraryFeature>(initialState: try w.current().state, run: f.runner.run)
        f.runner.register(substitute)
        do {
            _ = try await get(r.directory, r.registrationGeneration)
            throw VoxLeaseProbeError(description: "old store with a reused owner UUID acquired another lease")
        } catch let failure as VoxLeaseProbeError { throw failure } catch {}
        w.end()
        try await w.events.wait("usage-end")
        weak let lost = w.library
        w.drop()
        try requireLease(lost == nil, "acquirer and acquired token must not retain view store")
        do {
            _ = try await get(r.directory, r.registrationGeneration)
            throw VoxLeaseProbeError(description: "deallocated window acquired a new token")
        } catch let failure as VoxLeaseProbeError { throw failure } catch {}
        try await f.refuseDelete(r.identity)
        try await f.counts(r, physical: 1, displayed: 1)
        await token.release()
        try await f.counts(r, physical: 0, displayed: 0)
    }
}
