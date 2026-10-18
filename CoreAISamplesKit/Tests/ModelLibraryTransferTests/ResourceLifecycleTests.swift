import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import Testing

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ResourceLifecycleTests {
    @Test(
        "DL-04 An interrupted staging area is not installed after a restart, and only files that still match the pinned size and hash are reused"
    )
    func reusesVerifiedStaging() async throws {
        let sample = try Sample()
        let interrupted = try Harness(sample)
        defer { interrupted.close() }
        let corrupted = "token_embeddings.bf16"
        let resized = "tokenizer.json"
        let absent = "vision.aimodel/main.mlirb"
        var staged = sample.files
        staged[corrupted] = Data((staged[corrupted] ?? Data()).reversed())
        staged[resized]?.append(0)
        staged[absent] = nil
        staged[absent + ".part"] = Data("partial".utf8)
        staged["MODEL_LICENSE.txt.part"] = Data("partial".utf8)
        staged["._unexpected"] = Data("apple double".utf8)
        staged["text.aimodel/._unexpected"] = Data("apple double".utf8)
        staged["README.md"] = Data("extra".utf8)
        staged["notes/extra.txt"] = Data("extra".utf8)
        try write(
            staged, to: try await interrupted.store.stagingDirectory(for: sample.model, artifact: sample.artifact))

        let restarted = try Harness(sample, base: interrupted.base)
        defer { restarted.close() }
        let library = restarted.library()
        var events = restarted.events.makeAsyncIterator()
        library.send(.refresh)
        _ = try await next(&events, .ended("inspect"))
        #expect(library.state.installations[sample.id]?.phase == .missing)
        #expect(try await restarted.inspect().isMissing)

        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .ended("validate"))
        let refetched = [corrupted, resized, absent]
        #expect(restarted.io.requests.sorted() == refetched.map(sample.url).sorted())
        let reused = sample.artifact.files.filter { !refetched.contains($0.path) }.reduce(0) { $0 + $1.bytes }
        let first = restarted.issued.lazy.compactMap { entry -> ModelTransferProgress? in
            if case .progress(_, _, let progress) = entry.action { progress } else { nil }
        }.first
        #expect(first?.completedBytes == reused)
        let resource = try #require(library.state.installations[sample.id]?.resource)
        #expect(files(in: resource.directory) == Set(sample.files.keys))
    }

    @Test(
        "DL-04 AppleDouble leftovers in a verified staging area are removed, so the next download installs without new requests"
    )
    func removesAppleDoubleLeftovers() async throws {
        let harness = try Harness(try Sample())
        defer { harness.close() }
        let sample = harness.sample
        let staging = try await harness.store.stagingDirectory(for: sample.model, artifact: sample.artifact)
        try write(sample.files, to: staging)
        let leftovers = ["._unexpected", "text.aimodel/._unexpected"]
        try write(Dictionary(uniqueKeysWithValues: leftovers.map { ($0, Data("apple double".utf8)) }), to: staging)
        let library = harness.library()
        var events = harness.events.makeAsyncIterator()

        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .ended("validate"))
        let resource = try #require(library.state.installations[sample.id]?.resource)
        #expect(library.state.installations[sample.id]?.phase == .ready && harness.io.requests.isEmpty)
        #expect(files(in: resource.directory) == Set(sample.files.keys))
    }

    @Test("A failed migration of a matching legacy copy is reported as an operation failure instead of a missing model")
    func reportsFailedMigration() async throws {
        let sample = try Sample()
        let base = temporaryDirectory()
        let legacy = base.appending(path: "Documents/embeddinggemma2")
        try write(sample.files, to: legacy)
        let harness = try Harness(sample, base: base, candidates: [legacy])
        defer { harness.close() }
        let library = harness.library()
        var events = harness.events.makeAsyncIterator()
        harness.io.update { $0.capacity = 0 }

        library.send(.refresh)
        _ = try await next(&events, .ended("inspect"))
        let failed = try #require(library.state.installations[sample.id])
        #expect(failed.phase == .failed && failed.failure == .operation("insufficientSpace") && failed.resource == nil)
        #expect(try await harness.inspect().isMissing && files(in: legacy) == Set(sample.files.keys))

        harness.io.update { $0.capacity = .max }
        library.send(.refresh)
        _ = try await next(&events, .ended("inspect"))
        #expect(library.state.installations[sample.id]?.phase == .ready && harness.io.requests.isEmpty)
    }

    @Test(
        "Inspection migrates a matching legacy copy without downloading it and reports an invalid installed bundle as missing"
    )
    func inspectsAndMigrates() async throws {
        let sample = try Sample()
        let base = temporaryDirectory()
        let legacy = base.appending(path: "Documents/embeddinggemma2")
        try write(sample.files, to: legacy)
        let harness = try Harness(sample, base: base, candidates: [legacy])
        defer { harness.close() }
        let library = harness.library()
        var events = harness.events.makeAsyncIterator()

        library.send(.refresh)
        _ = try await next(&events, .ended("inspect"))
        let resource = try #require(library.state.installations[sample.id]?.resource)
        #expect(resource.matches(model: sample.model, artifact: sample.artifact) && harness.io.requests.isEmpty)
        #expect(files(in: legacy) == Set(sample.files.keys))

        try Data("changed".utf8).write(to: resource.directory.appending(path: "tokenizer.json"))
        library.send(.refresh)
        _ = try await next(&events, .ended("inspect"))
        #expect(library.state.installations[sample.id] == ModelInstallation())
        #expect(try await harness.inspect().resource == nil)
    }

    @Test(
        "Deletion is refused while the resource is leased and succeeds once it is released, and a demo release is reported as not connected"
    )
    func deletesAndRefusesDemoRelease() async throws {
        let harness = try Harness(try Sample())
        defer { harness.close() }
        let sample = harness.sample
        let library = harness.library(connected: true)
        var events = harness.events.makeAsyncIterator()
        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .ended("validate"))
        let resource = try #require(library.state.installations[sample.id]?.resource)

        let lease = await harness.store.lease(identity: resource.identity, owner: UUID())
        library.send(.deleteRequested(sample.id))
        _ = try await next(&events, .ended("delete"))
        #expect(library.state.installations[sample.id]?.failure == .modelInUse)
        #expect(try await harness.inspect().resource == resource)
        await lease.release()
        library.send(.refresh)
        _ = try await next(&events, .ended("inspect"))
        library.send(.deleteRequested(sample.id))
        _ = try await next(&events, .ended("delete"))
        #expect(library.state.installations[sample.id] == ModelInstallation())
        #expect(try await harness.inspect().isMissing)

        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .ended("validate"))
        library.send(.openDemo(sample.demo))
        let session = try #require(library.state.session)
        library.send(.closeDemo)
        _ = try await next(&events, .ended("stopDemo"))
        #expect(library.state.failure == .operation("Demo execution is not connected"))
        #expect(library.state.session == session && library.state.selection == sample.demo)
    }

    @Test(
        "DL-05 Downloading from the list or a demo description keeps the screen, search and selection, and completion never starts a demo by itself",
        arguments: [(false, false), (true, false), (true, true)])
    func keepsNavigation(fromDemo: Bool, connected: Bool) async throws {
        let harness = try Harness(try Sample())
        defer { harness.close() }
        let sample = harness.sample
        let library = harness.library(connected: connected)
        var events = harness.events.makeAsyncIterator()
        library.send(.queryChanged("image"))
        if fromDemo { library.send(.openDemo(sample.demo)) }
        let before = library.state

        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .ended("validate"))
        let after = library.state
        #expect(after.installations[sample.id]?.phase == .ready)
        #expect(after.query == before.query && after.selection == before.selection)
        #expect(after.selectedArtifacts == before.selectedArtifacts && after.sections == before.sections)
        #expect((after.session != nil) == connected && after.session?.selection == (connected ? sample.demo : nil))
        let downloaded = try #require(
            harness.issued.lazy.compactMap { entry -> (LibraryTicket, URL)? in
                if case .downloaded(_, let ticket, let staging) = entry.action { (ticket, staging) } else { nil }
            }.first)
        #expect(
            harness.issued.flatMap(\.commands) == [
                .run(.validate(sample.model, sample.artifact, downloaded.0, downloaded.1), id: .resource(sample.id))
            ])
    }
}
