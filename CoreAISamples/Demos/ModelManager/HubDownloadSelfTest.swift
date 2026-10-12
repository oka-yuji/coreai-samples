import CryptoKit
import Foundation
import Synchronization

enum HubDownloadSelfTest {
    struct Check {
        let name: String
        let passed: Bool
        let detail: String
    }

    private static let revision = "71034c5d8bde858ff824298bdedc65515b97d2b9"
    private static let repo = "hf-internal-testing/tiny-random-gpt2"
    private static let payloads = [
        "metadata.json": Data("metadata\n".utf8),
        "weights.bin": Data([0, 1, 2, 255]),
        "empty": Data(),
    ]
    private static let manifest = Data(
        """
        [
          {"type":"file","path":"metadata.json","size":9,"oid":"850a120dc4da9b38fceb1c17bec6ac21cb2aa6bc"},
          {"type":"file","path":"weights.bin","size":4,"lfs":{"size":4,"oid":"3d1f57c984978ef98a18378c8166c1cb8ede02c03eeb6aee7e2f121dfeee3e56"}},
          {"type":"file","path":"empty","size":0,"oid":"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"}
        ]
        """.utf8)

    private actor FakeHub {
        let manifest: Data
        let payloads: [String: Data]
        private(set) var downloads: [String] = []
        private(set) var lists = 0

        init(manifest: Data = HubDownloadSelfTest.manifest, payloads: [String: Data] = HubDownloadSelfTest.payloads) {
            self.manifest = manifest
            self.payloads = payloads
        }

        func list(_ url: URL) throws -> (Data, URLResponse) {
            lists += 1
            return (manifest, try response(url))
        }

        func file(_ url: URL, _ destination: URL) throws -> URLResponse {
            let path = url.lastPathComponent
            downloads.append(path)
            guard let data = payloads[path] else { throw HubDownloadError.invalidFile(path) }
            try data.write(to: destination)
            return try response(url)
        }

        private func response(_ url: URL) throws -> HTTPURLResponse {
            guard let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil) else {
                throw HubDownloadError.invalidURL(url.absoluteString)
            }
            return response
        }
    }

    private static func downloader(_ hub: FakeHub) -> HubDownloader {
        HubDownloader(
            listLoader: { try await hub.list($0) }, fileLoader: { try await hub.file($0, $1) },
            maximumConcurrentDownloads: 1)
    }

    static func run(offline: Bool) async -> [Check] {
        let root = FileManager.default.temporaryDirectory.appending(path: "hub-selftest-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        var checks: [Check] = []
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            checks += try await validAndResumed(in: root)
            checks += try await corruptedFiles(in: root)
            checks += try await refusedInputs(in: root)
            checks += try await cancellation(in: root)
            if !offline { checks += try await liveHub(in: root) }
        } catch {
            checks.append(Check(name: "hub/unexpected", passed: false, detail: error.localizedDescription))
        }
        return checks
    }

    private static func markerExists(_ directory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: directory.appending(path: ModelStorage.markerName).path(percentEncoded: false))
    }

    private static func sha256Files(in directory: URL, paths: [String]) throws -> [String: String] {
        try Dictionary(
            uniqueKeysWithValues: paths.map { path in
                let data = try Data(contentsOf: directory.appending(path: path))
                return (path, SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
            })
    }

    private static func validAndResumed(in root: URL) async throws -> [Check] {
        let baseline = root.appending(path: "complete")
        let initialHub = FakeHub()
        try await downloader(initialHub).download(repoID: repo, revision: revision, to: baseline)
        let paths = Array(payloads.keys).sorted()
        let hashes = try sha256Files(in: baseline, paths: paths)
        let marker = try JSONDecoder().decode(
            ModelStorage.CompletionMarker.self,
            from: Data(contentsOf: baseline.appending(path: ModelStorage.markerName)))
        let complete =
            marker.fileCount == 3 && marker.totalBytes == 13
            && hashes["metadata.json"] == "d99f5c6a7787cafbdb79e10d5eeca9777b43042cb0fcfa6c6be4f567f288913e"
            && hashes["weights.bin"] == "3d1f57c984978ef98a18378c8166c1cb8ede02c03eeb6aee7e2f121dfeee3e56"
        let repeatedHub = FakeHub()
        try await downloader(repeatedHub).download(repoID: repo, revision: revision, to: baseline)
        let skipped = await repeatedHub.downloads.isEmpty
        let partial = root.appending(path: "partial")
        let interruptedHub = FakeHub()
        let interrupted = downloader(interruptedHub)
        var cancelled = false
        do {
            try await interrupted.download(
                repoID: repo, revision: revision, to: partial,
                verified: { _, _ in
                    interrupted.cancel()
                })
        } catch HubDownloadError.cancelled { cancelled = true }
        let initialDownloads = await interruptedHub.downloads
        let noMarker = !markerExists(partial)
        let resumedHub = FakeHub()
        try await downloader(resumedHub).download(repoID: repo, revision: revision, to: partial)
        let resumed = await resumedHub.downloads
        let resumeMatches = try sha256Files(in: partial, paths: paths) == hashes
        return [
            Check(name: "hub/verifiedCompletion", passed: complete, detail: "\(hashes)"),
            Check(name: "hub/gitBlobSHA1AndEmptyFile", passed: complete, detail: "git and LFS fixtures verified"),
            Check(name: "hub/skipsVerifiedFiles", passed: skipped, detail: "a repeated run must download no files"),
            Check(
                name: "hub/fileResume",
                passed: cancelled && noMarker && initialDownloads == ["metadata.json"]
                    && Set(resumed) == ["weights.bin", "empty"] && resumeMatches,
                detail: "before=\(initialDownloads), after=\(resumed)"),
        ]
    }

    private static func corruptedFiles(in root: URL) async throws -> [Check] {
        let wrong = root.appending(path: "wrong-download")
        var badPayloads = payloads
        badPayloads["metadata.json"] = Data("Metadata\n".utf8)
        let corruptHub = FakeHub(payloads: badPayloads)
        var namedFailure = false
        do {
            try await downloader(corruptHub).download(repoID: repo, revision: revision, to: wrong)
        } catch HubDownloadError.checksumMismatch(let path) { namedFailure = path == "metadata.json" }
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: wrong.path(percentEncoded: false))
        let preexisting = root.appending(path: "wrong-existing")
        try FileManager.default.createDirectory(at: preexisting, withIntermediateDirectories: true)
        try Data("Metadata\n".utf8).write(to: preexisting.appending(path: "metadata.json"))
        ModelStorage.writeCompletionMarker(at: preexisting, fileCount: 3, totalBytes: 13)
        let repairedHub = FakeHub()
        try await downloader(repairedHub).download(repoID: repo, revision: revision, to: preexisting)
        let repaired = try Data(contentsOf: preexisting.appending(path: "metadata.json")) == payloads["metadata.json"]
        let requested = await repairedHub.downloads
        let failedExisting = root.appending(path: "wrong-existing-failure")
        try FileManager.default.createDirectory(at: failedExisting, withIntermediateDirectories: true)
        try Data("Metadata\n".utf8).write(to: failedExisting.appending(path: "metadata.json"))
        ModelStorage.writeCompletionMarker(at: failedExisting, fileCount: 3, totalBytes: 13)
        do {
            try await downloader(FakeHub(payloads: badPayloads)).download(
                repoID: repo, revision: revision, to: failedExisting)
        } catch HubDownloadError.checksumMismatch {}
        return [
            Check(
                name: "hub/removesMismatchedDownload",
                passed: namedFailure && leftovers.isEmpty && !markerExists(wrong), detail: "leftovers=\(leftovers)"),
            Check(
                name: "hub/redownloadsSameSizeCorruption",
                passed: repaired && requested.contains("metadata.json") && markerExists(preexisting),
                detail: "requested=\(requested)"),
            Check(
                name: "hub/removesStaleMarkerOnFailure", passed: !markerExists(failedExisting),
                detail: "a failed validation cannot retain a completion marker"),
        ]
    }

    private static func refusedInputs(in root: URL) async throws -> [Check] {
        let invalidLists = [
            "[{\"type\":\"file\",\"path\":\"metadata.json\",\"size\":9}]",
            "[{\"type\":\"file\",\"path\":\"weights.bin\",\"size\":4,\"lfs\":{\"oid\":\"bad\",\"size\":4}}]",
            "[{\"type\":\"file\",\"path\":\"../outside\",\"size\":0,\"oid\":\"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391\"}]",
            "[{\"type\":\"file\",\"path\":\".download-complete.json\",\"size\":0,\"oid\":\"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391\"}]",
        ]
        var allRefused = true
        for (index, list) in invalidLists.enumerated() {
            let hub = FakeHub(manifest: Data(list.utf8))
            var refused = false
            do {
                try await downloader(hub).download(
                    repoID: repo, revision: revision, to: root.appending(path: "invalid-\(index)"))
            } catch HubDownloadError.invalidFile { refused = true }
            let fetched = await hub.downloads
            allRefused = allRefused && refused && fetched.isEmpty
        }
        var revisionsRefused = true
        for value in ["main", "v1", String(revision.prefix(39)), String(repeating: "z", count: 40)] {
            let model = CatalogModel(
                id: "fixture", name: "Fixture", repo: repo, revision: value, sizeBytes: 13, memory: "",
                platforms: "macOS")
            let hub = FakeHub()
            var refused = false
            do {
                try await downloader(hub).download(
                    repoID: model.repo, revision: model.revision, to: root.appending(path: "revision"))
            } catch HubDownloadError.invalidRevision { refused = true }
            let lists = await hub.lists
            revisionsRefused = revisionsRefused && refused && lists == 0
        }
        let outside = root.appending(path: "outside")
        try Data("untouched".utf8).write(to: outside)
        let symlinked = root.appending(path: "symlinked")
        try FileManager.default.createDirectory(at: symlinked, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: symlinked.appending(path: "metadata.json"), withDestinationURL: outside)
        var symlinkRefused = false
        do {
            try await downloader(FakeHub()).download(repoID: repo, revision: revision, to: symlinked)
        } catch HubDownloadError.invalidFile { symlinkRefused = true }
        let outsideIntact = try Data(contentsOf: outside) == Data("untouched".utf8)
        return [
            Check(
                name: "hub/rejectsUnverifiableManifest", passed: allRefused,
                detail: "no file I/O request for missing checksums or invalid paths"),
            Check(
                name: "hub/requiresPinnedRevision", passed: revisionsRefused,
                detail: "no listing request for non-commit revisions"),
            Check(
                name: "hub/refusesDestinationEscape", passed: symlinkRefused && outsideIntact,
                detail: "a destination symlink cannot escape its root"),
        ]
    }

    private static func cancellation(in root: URL) async throws -> [Check] {
        let beforeHub = FakeHub()
        let before = downloader(beforeHub)
        before.cancel()
        var beforeRefused = false
        do {
            try await before.download(repoID: repo, revision: revision, to: root.appending(path: "cancel-before"))
        } catch HubDownloadError.cancelled { beforeRefused = true }
        let beforeLists = await beforeHub.lists
        let last = root.appending(path: "cancel-last")
        let lastDownloader = downloader(FakeHub())
        var lastRefused = false
        do {
            try await lastDownloader.download(
                repoID: repo, revision: revision, to: last,
                verified: { path, _ in
                    if path == "empty" { lastDownloader.cancel() }
                })
        } catch HubDownloadError.cancelled { lastRefused = true }
        let bytes = root.appending(path: "large-for-hash")
        try Data(repeating: 42, count: 4_194_304).write(to: bytes)
        let (hashRefused, hashCalls, hashRanOnMain) = await cancelHashFromMainActor(at: bytes)
        let duringTransfer = root.appending(path: "cancel-transfer")
        let transfer = downloader(FakeHub())
        var transferRefused = false
        do {
            try await transfer.download(
                repoID: repo, revision: revision, to: duringTransfer,
                progress: { _ in
                    transfer.cancel()
                })
        } catch HubDownloadError.cancelled { transferRefused = true }
        let transferFiles = try FileManager.default.contentsOfDirectory(
            atPath: duringTransfer.path(percentEncoded: false))
        let taskHub = FakeHub()
        let taskDownloader = downloader(taskHub)
        let taskRefused = try await withThrowingTaskGroup(of: Bool.self) { group in
            group.cancelAll()
            group.addTask {
                do {
                    try await taskDownloader.download(
                        repoID: repo, revision: revision, to: root.appending(path: "cancel-task"))
                    return false
                } catch HubDownloadError.cancelled { return true }
            }
            return try await group.next() ?? false
        }
        let taskLists = await taskHub.lists
        return [
            Check(
                name: "hub/cancelBeforeRegistration", passed: beforeRefused && beforeLists == 0,
                detail: "cancellation starts no request"),
            Check(
                name: "hub/cancelBeforeCompletionMarker", passed: lastRefused && !markerExists(last),
                detail: "cancellation after the last verification writes no marker"),
            Check(
                name: "hub/cancelDuringHash", passed: hashRefused && hashCalls == 3,
                detail: "checks=\(hashCalls)"),
            Check(
                name: "hub/hashesOffMainThread", passed: !hashRanOnMain,
                detail: "hashing runs on the concurrent executor"),
            Check(
                name: "hub/taskCancellation", passed: taskRefused && taskLists == 0,
                detail: "task cancellation propagates before a request begins"),
            Check(
                name: "hub/cancelAfterTransfer",
                passed: transferRefused && transferFiles.isEmpty && !markerExists(duringTransfer),
                detail: "cancelling after transfer removes the unverified temporary file"),
        ]
    }

    @MainActor
    private static func cancelHashFromMainActor(at bytes: URL) async -> (Bool, Int, Bool) {
        let state = Mutex((calls: 0, ranOnMain: false))
        var hashRefused = false
        do {
            _ = try await HubDownloader.fileDigest(at: bytes, checksum: .sha256("unused"), size: 4_194_304) {
                state.withLock { value in
                    value.calls += 1
                    value.ranOnMain = value.ranOnMain || Thread.isMainThread
                    return value.calls == 3
                }
            }
        } catch HubDownloadError.cancelled { hashRefused = true } catch {
            return (false, 0, true)
        }
        let hashState = state.withLock { $0 }
        return (hashRefused, hashState.calls, hashState.ranOnMain)
    }

    private static func liveHub(in root: URL) async throws -> [Check] {
        let directory = root.appending(path: "live")
        let first = HubDownloader(maximumConcurrentDownloads: 1)
        let firstEvents = Mutex([(String, Bool)]())
        try await first.download(
            repoID: repo, revision: revision, to: directory,
            verified: { path, skipped in
                firstEvents.withLock { $0.append((path, skipped)) }
            })
        let events = firstEvents.withLock { $0 }
        let paths = events.map(\.0)
        let hashes = try sha256Files(in: directory, paths: paths)
        let marker = try JSONDecoder().decode(
            ModelStorage.CompletionMarker.self,
            from: Data(contentsOf: directory.appending(path: ModelStorage.markerName)))
        let repeatedEvents = Mutex([(String, Bool)]())
        try await HubDownloader().download(
            repoID: repo, revision: revision, to: directory,
            verified: { path, skipped in
                repeatedEvents.withLock { $0.append((path, skipped)) }
            })
        let repeated = repeatedEvents.withLock { $0 }
        let partial = root.appending(path: "live-partial")
        let stopped = HubDownloader(maximumConcurrentDownloads: 1)
        let stoppedEvents = Mutex([(String, Bool)]())
        var cancelled = false
        do {
            try await stopped.download(
                repoID: repo, revision: revision, to: partial,
                verified: { path, skipped in
                    stoppedEvents.withLock { $0.append((path, skipped)) }
                    stopped.cancel()
                })
        } catch HubDownloadError.cancelled { cancelled = true }
        let before = stoppedEvents.withLock { $0 }
        let partialHadNoMarker = !markerExists(partial)
        let resumeEvents = Mutex([(String, Bool)]())
        try await HubDownloader().download(
            repoID: repo, revision: revision, to: partial,
            verified: { path, skipped in
                resumeEvents.withLock { $0.append((path, skipped)) }
            })
        let resumed = resumeEvents.withLock { $0 }
        let afterHashes = try sha256Files(in: partial, paths: paths)
        return [
            Check(
                name: "hub/liveVerified",
                passed: paths.count == 10 && marker.fileCount == 10 && marker.totalBytes == 12_512_611
                    && events.allSatisfy { !$0.1 },
                detail: "files=\(paths.count), bytes=\(marker.totalBytes), sha256=\(hashes)"),
            Check(
                name: "hub/liveSkipsAllVerifiedFiles", passed: repeated.count == 10 && repeated.allSatisfy(\.1),
                detail: "verified skips=\(repeated.count)"),
            Check(
                name: "hub/liveResumesAfterFirstFile",
                passed: cancelled && partialHadNoMarker && before.count == 1
                    && resumed.filter(\.1).map(\.0) == before.map(\.0) && resumed.filter { !$0.1 }.count == 9
                    && hashes == afterHashes,
                detail: "before=\(before.map(\.0)), skipped=\(resumed.filter(\.1).map(\.0)), sha256=\(afterHashes)"),
        ]
    }
}
