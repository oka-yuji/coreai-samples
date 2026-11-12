import Foundation
import ModelLibraryFeature
import ModelLibraryTransfer
import Synchronization
import Testing

@Suite(.timeLimit(.minutes(1)))
struct ArtifactTransferTests {
    private let names = ["a.bin", "b c.bin", "dir/#?.bin", "dir/d.bin", "e.bin"]

    private func sample(root: String = "release/v1", pinned: String = revision) throws -> Sample {
        try Sample(
            root: root, files: Dictionary(uniqueKeysWithValues: names.map { ($0, Data("file \($0)".utf8)) }),
            contract: "fixture", pinned: pinned)
    }

    @Test("CD-01 A manifest download uses the remote commit and root and reuses completed files on retry")
    func manifestLocator() async throws {
        let data = Data("manifest fixture".utf8)
        let sample = try manifestSample(data: data)
        let (_, sink) = AsyncStream.makeStream(of: Event.self)
        let io = FakeIO(sink: sink)
        let staging = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: staging) }
        let expected = "https://huggingface.co/owner/fixture/resolve/\(revision)/bundle/bundle.json"
        io.update { $0.replies[expected] = .content(data) }
        let transfer = ArtifactTransfer(fileLoader: io.load, availableCapacity: { _ in .max })
        for _ in 0..<2 {
            try await transfer.download(model: sample.model, artifact: sample.artifact, to: staging) { _ in }
        }
        #expect(io.requests == [expected])
        #expect(io.sentRequests.allSatisfy { ($0.allHTTPHeaderFields ?? [:]).isEmpty && $0.url?.query == nil })
        #expect(try Data(contentsOf: staging.appending(path: "bundle.json")) == data)
        #expect(sample.artifact.sourceKind == .localManifest && sample.artifact.sourceRevision == sha256(data))
    }

    @Test(
        "CD-01 Invalid manifests and overflowing sizes are refused before touching staging or making a request",
        arguments: [false, true])
    func invalidManifestTransfer(overflow: Bool) async throws {
        let data = Data("manifest fixture".utf8)
        let sample = try manifestSample(data: data, invalidManifest: !overflow, overflow: overflow)
        let (_, sink) = AsyncStream.makeStream(of: Event.self)
        let io = FakeIO(sink: sink)
        let staging = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: staging) }
        let transfer = ArtifactTransfer(fileLoader: io.load, availableCapacity: { _ in .max })
        let error = await #expect(throws: ArtifactTransferError.self) {
            try await transfer.download(model: sample.model, artifact: sample.artifact, to: staging) { _ in }
        }
        guard case .invalidDefinition? = error else {
            Issue.record("Expected invalid definition")
            return
        }
        #expect(io.requests.isEmpty && !FileManager.default.fileExists(atPath: staging.path))
    }

    private func manifestSample(data: Data, invalidManifest: Bool = false, overflow: Bool = false) throws -> (
        model: LibraryModel, artifact: ModelArtifact
    ) {
        let original = try Sample(root: "bundle", files: ["bundle.json": data], contract: "fixture")
        var object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(original.artifact)) as? [String: Any])
        object["localBundle"] = [
            "manifestSHA256": invalidManifest ? String(repeating: "a", count: 64) : sha256(data),
            "profileID": "fixture",
        ]
        if overflow {
            object["files"] = [
                ["path": "bundle.json", "bytes": Int64.max, "sha256": sha256(data)],
                ["path": "overflow.bin", "bytes": 1, "sha256": sha256(data)],
            ]
        }
        let artifact = try JSONDecoder().decode(
            ModelArtifact.self, from: JSONSerialization.data(withJSONObject: object))
        let model = LibraryModel(
            id: "fixture", name: "Fixture", category: "Forecasting", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.iOS], demos: [])
        return (model, artifact)
    }

    @Test(
        "DL-06 Each pinned file is requested from its encoded resolve URL without credentials or a query, at most three at a time in definition order"
    )
    func requestsFilesInOrder() async throws {
        let sample = try sample()
        let (events, sink) = AsyncStream.makeStream(of: Event.self)
        let io = FakeIO(sink: sink)
        let staging = temporaryDirectory()
        defer {
            io.releaseAll()
            try? FileManager.default.removeItem(at: staging)
        }
        let urls = ["a.bin", "b%20c.bin", "dir/%23%3F.bin", "dir/d.bin", "e.bin"].map {
            "https://huggingface.co/owner/fixture/resolve/\(revision)/release/v1/" + $0
        }
        io.update { state in
            for (name, url) in zip(names, urls) { state.replies[url] = .held(sample.files[name] ?? Data()) }
        }
        let reports = Mutex<[Int64]>([])
        let transfer = ArtifactTransfer(fileLoader: io.load, availableCapacity: { _ in .max })
        let download = Task {
            try await transfer.download(model: sample.model, artifact: sample.artifact, to: staging) { progress in
                reports.withLock { $0.append(progress.completedBytes) }
            }
        }
        var iterator = events.makeAsyncIterator()
        _ = try await next(&iterator, all: urls.prefix(3).map { .started($0) })
        #expect(Set(io.requests) == Set(urls.prefix(3)) && io.peak == 3)
        for (index, url) in urls.enumerated() {
            io.release(url)
            if index + 3 < urls.count { _ = try await next(&iterator, .started(urls[index + 3])) }
        }
        try await download.value
        #expect(io.requests.dropFirst(3) == urls.dropFirst(3) && io.peak == 3)
        #expect(
            io.sentRequests.allSatisfy {
                $0.httpMethod == "GET" && ($0.allHTTPHeaderFields ?? [:]).isEmpty && $0.url?.query == nil
            })
        let completed = reports.withLock { $0 }
        #expect(completed.first == 0 && completed.last == sample.total && completed == completed.sorted())
        #expect(Set(completed).count == completed.count && completed.count == urls.count + 1)
        #expect(files(in: staging) == Set(names))
    }

    @Test(
        "DL-06 Cancelling a transfer closes every request in flight before CancellationError is thrown and keeps the verified files"
    )
    func joinsCancelledRequests() async throws {
        let sample = try sample(root: "")
        let (events, sink) = AsyncStream.makeStream(of: Event.self)
        let io = FakeIO(sink: sink)
        let staging = temporaryDirectory()
        defer {
            io.releaseAll()
            try? FileManager.default.removeItem(at: staging)
        }
        let held = names.dropFirst(2).map(sample.url)
        io.update { state in
            for name in names {
                let data = sample.files[name] ?? Data()
                state.replies[sample.url(name)] = held.contains(sample.url(name)) ? .held(data) : .content(data)
            }
        }
        let transfer = ArtifactTransfer(fileLoader: io.load, availableCapacity: { _ in .max })
        let download = Task {
            defer { sink.yield(.ended("transfer")) }
            try await transfer.download(model: sample.model, artifact: sample.artifact, to: staging) { _ in }
        }
        var iterator = events.makeAsyncIterator()
        _ = try await next(&iterator, all: held.map { .started($0) })
        download.cancel()
        for url in held { io.release(url) }
        let seen = try await next(&iterator, .ended("transfer"))
        #expect(held.allSatisfy { precedes(.cancelled($0), .closed($0), in: seen) })
        #expect(held.allSatisfy { precedes(.closed($0), .ended("transfer"), in: seen) })
        await #expect(throws: CancellationError.self) { try await download.value }
        #expect(files(in: staging) == Set(names.prefix(2)))
    }

    @Test(
        "DL-06 Failed responses report only the file path and HTTP status and leave no partial file",
        arguments: [
            FakeIO.Reply.status(503), .disconnect, .content(Data("short".utf8)), .content(Data("file e.biN".utf8)),
        ])
    func reportsFailures(_ reply: FakeIO.Reply) async throws {
        let sample = try sample(root: "")
        let (_, sink) = AsyncStream.makeStream(of: Event.self)
        let io = FakeIO(sink: sink)
        let staging = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: staging) }
        io.update { state in
            for name in names { state.replies[sample.url(name)] = .content(sample.files[name] ?? Data()) }
            state.replies[sample.url("e.bin")] = reply
        }
        let transfer = ArtifactTransfer(fileLoader: io.load, availableCapacity: { _ in .max })
        let error = await #expect(throws: ArtifactTransferError.self) {
            try await transfer.download(model: sample.model, artifact: sample.artifact, to: staging) { _ in }
        }
        let expected = sample.files["e.bin"]?.count
        switch (reply, error) {
        case (.status, .http(503, "e.bin")?): break
        case (.content(let data), .size("e.bin")?): #expect(data.count != expected)
        case (.content(let data), .checksum("e.bin")?): #expect(data.count == expected)
        case (.disconnect, .transport("e.bin", let underlying)?):
            #expect((underlying as? URLError)?.code == .networkConnectionLost)
            #expect((underlying as NSError).userInfo.isEmpty)
        default: Issue.record("Unexpected failure \(String(describing: error)) for \(reply)")
        }
        #expect(files(in: staging).isSubset(of: Set(names)) && !files(in: staging).contains("e.bin"))
    }

    @Test(
        "DL-06 Definitions with an unpinned revision, a path outside the staging area or a path that collides with a partial file are refused before any request",
        arguments: [
            (["a.bin"], "main"), (["../escape.bin"], revision), (["dir/./a.bin"], revision), (["/a.bin"], revision),
            (["model.bin", "model.bin.part"], revision),
        ])
    func refusesUnsafeDefinitions(paths: [String], pinned: String) async throws {
        let sample = try Sample(
            files: Dictionary(uniqueKeysWithValues: paths.map { ($0, Data($0.utf8)) }), contract: "fixture",
            pinned: pinned)
        let (_, sink) = AsyncStream.makeStream(of: Event.self)
        let io = FakeIO(sink: sink)
        let base = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: base) }
        let transfer = ArtifactTransfer(fileLoader: io.load, availableCapacity: { _ in .max })
        let error = await #expect(throws: ArtifactTransferError.self) {
            try await transfer.download(
                model: sample.model, artifact: sample.artifact, to: base.appending(path: "staging")
            ) { _ in }
        }
        #expect(io.requests.isEmpty && files(in: base).isEmpty)
        guard case .invalidDefinition? = error else {
            Issue.record("Expected an invalid definition, got \(String(describing: error))")
            return
        }
    }
}
