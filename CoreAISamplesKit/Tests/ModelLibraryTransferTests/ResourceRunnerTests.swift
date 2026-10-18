import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import Testing

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ResourceRunnerTests {
    enum Failure: CaseIterable, Sendable {
        case http, disconnect, capacity, checksum, size, installation
    }

    @Test(
        "DL-01 Only the pinned files are fetched into staging, and the artifact becomes ready only after verification and installation succeed"
    )
    func fetchesPinnedFiles() async throws {
        let harness = try Harness(try Sample(root: "bundles/fp16"))
        defer { harness.close() }
        let sample = harness.sample
        let library = harness.library()
        var events = harness.events.makeAsyncIterator()
        let held = sample.url("vision.aimodel/main.mlirb")
        let data = try #require(sample.files["vision.aimodel/main.mlirb"])
        harness.io.update { $0.replies[held] = .held(data) }

        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .started(held))
        #expect(library.state.installations[sample.id]?.phase.isDownloading == true)
        #expect(try await harness.inspect().isMissing)
        harness.io.release(held)
        _ = try await next(&events, .ended("validate"))

        let installation = try #require(library.state.installations[sample.id])
        let resource = try #require(installation.resource)
        #expect(installation.phase == .ready && resource.matches(model: sample.model, artifact: sample.artifact))
        #expect(try await harness.inspect().resource == resource)
        #expect(files(in: resource.directory) == Set(sample.files.keys))
        #expect(!FileManager.default.fileExists(atPath: harness.staging.path))
        #expect(harness.io.requests.sorted() == sample.files.keys.map(sample.url).sorted())
        #expect(
            harness.io.sentRequests.allSatisfy {
                $0.httpMethod == "GET" && ($0.allHTTPHeaderFields ?? [:]).isEmpty && $0.url?.query == nil
            })
        let progress = harness.issued.compactMap { entry -> ModelTransferProgress? in
            if case .progress(_, _, let progress) = entry.action { progress } else { nil }
        }
        #expect(zip(progress, progress.dropFirst()).allSatisfy { $0.completedBytes < $1.completedBytes })
        #expect(progress.last == ModelTransferProgress(completedBytes: sample.total, totalBytes: sample.total))
    }

    @Test(
        "DL-02 Duplicate requests, requests while stopping and stale or foreign results are rejected, and the acknowledgement waits for the transfer to close"
    )
    func rejectsDuplicatesAndJoinsTransfers() async throws {
        let harness = try Harness(try Sample())
        defer { harness.close() }
        let sample = harness.sample
        let id = sample.id
        let library = harness.library()
        let other = harness.library()
        var events = harness.events.makeAsyncIterator()
        let held = sample.url("tokenizer.json")
        harness.io.update { $0.replies[held] = .held(sample.files["tokenizer.json"] ?? Data()) }

        library.send(.downloadRequested(id))
        _ = try await next(&events, .started(held))
        let first = try #require(library.state.installations[id]?.requestID)
        library.send(.downloadRequested(id))
        #expect(library.state.installations[id]?.requestID == first)
        other.send(.downloadRequested(id))
        _ = try await next(&events, .ended("download"))
        #expect(other.state.installations[id]?.phase == .failed)
        #expect(other.state.installations[id]?.failure == .operation("transfer in progress"))

        library.send(.cancelRequested(id))
        let cancelling = try await next(&events, .cancelled(held), .began("stopTransfer"))
        #expect(library.state.installations[id]?.phase == .stopping)
        library.send(.downloadRequested(id))
        #expect(library.state.installations[id]?.phase == .stopping)
        harness.io.release(held)
        let stopping = cancelling + (try await next(&events, .ended("stopTransfer")))
        #expect(precedes(.cancelled(held), .closed(held), in: stopping))
        #expect(precedes(.closed(held), .sent("transferStopped"), in: stopping))
        #expect(sent(Array(stopping.drop { $0 != .cancelled(held) })) == [.sent("transferStopped")])
        #expect(library.state.installations[id] == ModelInstallation())

        library.send(.downloadRequested(id))
        _ = try await next(&events, .started(held))
        let second = try #require(library.state.installations[id]?.requestID)
        #expect(second != first)
        let before = library.state
        let complete = try #require(ModelTransferProgress(completedBytes: sample.total, totalBytes: sample.total))
        let resource = ValidatedModelResource(
            modelID: sample.model.id, artifactID: id, contract: sample.artifact.contract, revision: revision,
            upstreamRevision: upstream.revision, directory: harness.staging, sizeBytes: sample.total)
        for ticket in [first, LibraryTicket(owner: UUID(), number: second.number)] {
            library.send(.progress(id, ticket, complete))
            library.send(.downloaded(id, ticket, harness.staging))
            library.send(.validated(id, ticket, resource))
            library.send(.failed(id, ticket, .invalidResource))
            library.send(.transferStopped(id, ticket))
        }
        #expect(library.state == before)
        harness.io.release(held)
        _ = try await next(&events, .ended("validate"))
        #expect(library.state.installations[id]?.phase == .ready)
        #expect(harness.io.requests.filter { $0 == held }.count == 2)
    }

    @Test("DL-02 Cancelling during validation is acknowledged only after the installation step has finished")
    func joinsValidation() async throws {
        let harness = try Harness(try Sample())
        defer { harness.close() }
        let sample = harness.sample
        let library = harness.library()
        var events = harness.events.makeAsyncIterator()
        harness.io.update { $0.ledger = .hold }

        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .ledger)
        #expect(library.state.installations[sample.id]?.phase == .validating)
        library.send(.cancelRequested(sample.id))
        _ = try await next(&events, .began("stopTransfer"))
        #expect(library.state.installations[sample.id]?.phase == .stopping)
        harness.io.releaseLedger()
        let stopping = try await next(&events, .ended("stopTransfer"))
        #expect(precedes(.ended("validate"), .sent("transferStopped"), in: stopping))
        #expect(sent(stopping) == [.sent("transferStopped")])
        #expect(library.state.installations[sample.id] == ModelInstallation())

        library.send(.refresh)
        _ = try await next(&events, .ended("inspect"))
        let resource = try #require(try await harness.inspect().resource)
        #expect(library.state.installations[sample.id]?.resource == resource)
    }

    @Test(
        "DL-03 HTTP, disconnection, capacity, checksum, size and installation failures make nothing ready, and the same model and demo can be retried",
        arguments: Failure.allCases)
    func recoversFromFailures(_ failure: Failure) async throws {
        let harness = try Harness(try Sample())
        defer { harness.close() }
        let sample = harness.sample
        let library = harness.library(connected: true)
        var events = harness.events.makeAsyncIterator()
        let path = "text.aimodel/main.mlirb"
        let url = sample.url(path)
        let data = try #require(sample.files[path])
        let expected: String
        switch failure {
        case .http:
            harness.io.update { $0.replies[url] = .status(404) }
            expected = ArtifactTransferError.http(404, path).localizedDescription
        case .disconnect:
            harness.io.update { $0.replies[url] = .disconnect }
            expected = ArtifactTransferError.transport(path, URLError(.networkConnectionLost)).localizedDescription
        case .capacity:
            harness.io.update { $0.capacity = sample.total - 1 }
            expected = ArtifactTransferError.insufficientSpace.localizedDescription
        case .checksum:
            harness.io.update { $0.replies[url] = .content(Data(data.reversed())) }
            expected = ArtifactTransferError.checksum(path).localizedDescription
        case .size:
            harness.io.update { $0.replies[url] = .content(data + Data("!".utf8)) }
            expected = ArtifactTransferError.size(path).localizedDescription
        case .installation:
            harness.io.update { $0.ledger = .fail }
            expected = CocoaError(.fileWriteOutOfSpace).localizedDescription
        }

        library.send(.openDemo(sample.demo))
        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .ended(failure == .installation ? "validate" : "download"))
        let failed = try #require(library.state.installations[sample.id])
        #expect(failed.phase == .failed && failed.failure == .operation(expected) && failed.resource == nil)
        #expect(library.state.selection == sample.demo && library.state.session == nil)
        #expect(try await harness.inspect().isMissing)
        #expect(files(in: harness.staging).isSubset(of: Set(sample.files.keys)))
        if failure == .capacity { #expect(harness.io.requests.isEmpty) }

        harness.io.update { state in
            state.replies[url] = .content(data)
            state.capacity = .max
            state.ledger = .write
        }
        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .ended("validate"))
        let session = try #require(library.state.session)
        #expect(library.state.installations[sample.id]?.phase == .ready)
        #expect(session.selection == sample.demo && library.state.access(to: sample.demo) == .ready(session))
    }

    @Test("DL-03 Cancelling as the last file completes makes nothing ready, and the retry fetches only that file")
    func cancelsAtCompletion() async throws {
        let harness = try Harness(try Sample())
        defer { harness.close() }
        let sample = harness.sample
        let library = harness.library(connected: true)
        var events = harness.events.makeAsyncIterator()
        let last = try #require(sample.artifact.files.last)
        let url = sample.url(last.path)
        let data = try #require(sample.files[last.path])
        harness.io.update { $0.replies[url] = .held(data) }
        let remaining = ModelTransferProgress(completedBytes: sample.total - last.bytes, totalBytes: sample.total)

        library.send(.openDemo(sample.demo))
        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .started(url))
        while library.state.installations[sample.id]?.phase != .downloading(remaining) {
            _ = try await next(&events, .sent("progress"))
        }
        library.send(.cancelRequested(sample.id))
        harness.io.release(url)
        _ = try await next(&events, .ended("stopTransfer"))
        #expect(library.state.installations[sample.id] == ModelInstallation())
        #expect(library.state.session == nil && library.state.selection == sample.demo)
        #expect(try await harness.inspect().isMissing)
        #expect(files(in: harness.staging) == Set(sample.files.keys).subtracting([last.path]))

        harness.io.update { state in
            state.replies[url] = .content(data)
            state.requests = []
        }
        library.send(.downloadRequested(sample.id))
        _ = try await next(&events, .ended("validate"))
        #expect(harness.io.requests == [url])
        #expect(library.state.installations[sample.id]?.phase == .ready)
        #expect(library.state.session?.selection == sample.demo)
    }

}
