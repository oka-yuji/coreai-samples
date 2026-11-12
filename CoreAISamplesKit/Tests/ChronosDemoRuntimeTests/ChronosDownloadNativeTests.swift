import ChronosCore
import ChronosFeature
import CryptoKit
import Darwin
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import SwiftUI
import Synchronization
import Testing

@testable import ChronosDemoRuntime
@testable import ModelLibraryRuntime

@MainActor
struct ChronosDownloadNativeTests {
    @Test(
        "CD-05 Fresh Models downloads match all fixed native outputs and preserve the original bundle after release",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_CHRONOS2_DOWNLOAD_OUTPUT"] != nil),
        .timeLimit(.minutes(30)))
    func freshModelsDownload() async throws {
        let environment = ProcessInfo.processInfo.environment
        let output = try newOutput(try #require(environment["COREAI_CHRONOS2_DOWNLOAD_OUTPUT"]))
        let source = URL(filePath: try #require(environment["COREAI_CHRONOS2_DOWNLOAD_SOURCE"]))
            .resolvingSymlinksInPath()
        let sourcePath = source.path(percentEncoded: false)
        let outputPath = output.path(percentEncoded: false)
        try #require(sourcePath != outputPath && !outputPath.hasPrefix(sourcePath + "/"))
        try #require(!sourcePath.hasPrefix(outputPath + "/"))
        let referenceData = try Data(
            contentsOf: URL(filePath: try #require(environment["COREAI_CHRONOS2_DOWNLOAD_REFERENCE"])))
        let referenceHash = NYCAPIContract.hash(referenceData)
        try #require(referenceHash == "8f894aaac769294a3e6843225e6f86519f85090d747b16d94a288fb987661470")
        let reference = try JSONDecoder().decode(ChronosScenesReference.self, from: referenceData)
        try #require(
            reference.format == "chronos2-nyc-reference/1" && reference.modelRevision == ChronosContract.revision
                && reference.atol == 0.0001 && reference.rtol == 0.0001 && reference.cases.count == 60)
        try #require(reference.sourceHashes == ChronosRepository.hashes.filter { $0.key != "forecasts.json" })
        let originalCatalog = try MacDemoCatalog.make()
        let model = try #require(originalCatalog.model("chronos2"))
        let demo = try #require(originalCatalog.demo(ChronosDemoCatalog.selection.demoID))
        let catalog = try ModelCatalog(models: [model], demos: [demo])
        let artifact = try ChronosModelArtifact.make()
        let download = try #require(artifact.download)
        try #require(
            download.repository == "okayuji/chronos-2-coreai-fp32"
                && download.revision == "d3ead488046498faa8b3b5103b1923e52a313d0e" && download.root == "bundle")
        let expectedFiles = artifact.files.sorted { $0.path < $1.path }
        let originalFiles = try await inventory(source, files: expectedFiles)
        try #require(originalFiles == expectedFiles && expectedFiles.count == 9)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let models = output.appending(path: "models")
        try #require(!FileManager.default.fileExists(atPath: models.path))
        let storage = ManagedModelStore(root: models)
        let prefix = "https://huggingface.co/\(download.repository)/resolve/\(download.revision)/bundle/"
        let expectedRequests = Set(expectedFiles.map { prefix + $0.path })
        let responses = Mutex<[String: Int]>([:])
        let transfer = ArtifactTransfer(fileLoader: { request, destination in
            let address = try #require(request.url?.absoluteString)
            try #require(expectedRequests.contains(address) && (request.allHTTPHeaderFields ?? [:]).isEmpty)
            let response = try await ArtifactTransfer.load(request, to: destination)
            let status = try #require((response as? HTTPURLResponse)?.statusCode)
            responses.withLock { $0[address] = status }
            return response
        })
        let adapter = try ChronosNativeAdapter(
            directory: ChronosRepository.bundledDirectory(),
            journal: ChronosJournalAssembly(files: ChronosJournalFiles(directory: output.appending(path: "journal"))))
        let assembly = ModelLibraryAssembly(
            adapters: [adapter], store: storage, transfer: transfer, migrationCandidates: { [] })
        let events = AsyncStream<ModelLibraryEffect>.makeStream()
        defer {
            events.continuation.finish()
            adapter.actions.continuation.finish()
            adapter.journalActions.continuation.finish()
        }
        let library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(
                catalog: catalog, environment: LibraryEnvironment(platform: .macOS, connectedDemos: [adapter.demoID]),
                owner: UUID())
        ) { effect, send in
            await assembly.runner.run(effect, send: send)
            events.continuation.yield(effect)
        }
        assembly.runner.register(library)
        var completed = events.stream.makeAsyncIterator()
        try write(
            ["format": "chronos2-download-checkpoint/1", "stage": "before-download"], to: output,
            name: "checkpoint.json")
        do {
            library.send(.refresh)
            _ = try #require(await completed.next(isolation: #isolation))
            try #require(library.state.installations[artifact.id]?.phase == .missing && library.state.session == nil)
            library.send(.downloadRequested(artifact.id))
            while let effect = await completed.next(isolation: #isolation) {
                if case .validate = effect { break }
                if library.state.installations[artifact.id]?.phase == .failed { break }
            }
            try #require(library.state.installations[artifact.id]?.phase == .ready)
            let installed = try #require(library.state.installations[artifact.id]?.resource)
            try #require(
                installed.sourceKind == .localManifest && installed.revision == ChronosModelArtifact.manifestSHA256)
            try #require(installed.matches(model: model, artifact: artifact))
            try #require(installed.sizeBytes == 478_864_956 && library.state.session == nil && adapter.store == nil)
            let requests = responses.withLock { $0 }
            try #require(Set(requests.keys) == expectedRequests && requests.values.allSatisfy { $0 == 200 })
            try #require(try await inventory(installed.directory, files: expectedFiles) == expectedFiles)
            let relative = installed.directory.path(percentEncoded: false)
            try #require(relative.hasPrefix(outputPath + "/"))
            try write(
                [
                    "format": "chronos2-download-acquisition/1", "repository": download.repository,
                    "revision": download.revision, "root": download.root, "requests": requests,
                    "files": fileRows(expectedFiles), "bytes": installed.sizeBytes,
                    "sourceKind": installed.sourceKind.rawValue, "manifestSHA256": installed.revision,
                    "managedRelativePath": String(relative.dropFirst(outputPath.count + 1)),
                    "freshManagedStore": true, "migrationCandidates": 0,
                ], to: output, name: "acquisition.json")
            let forecasts = try await predict(
                library: library, assembly: assembly, adapter: adapter, completed: &completed,
                installed: installed, reference: reference, storage: storage)
            try #require(try await inventory(source, files: expectedFiles) == originalFiles)
            try write(
                [
                    "format": "chronos2-download-native/1", "repository": download.repository,
                    "downloadRevision": download.revision, "sourceKind": installed.sourceKind.rawValue,
                    "manifestSHA256": installed.revision, "referenceSHA256": referenceHash,
                    "cases": 60, "elements": 20_160, "atol": 0.0001, "rtol": 0.0001,
                    "forecasts": forecasts, "originalSourceUnchanged": true, "leasesAfterClose": 0,
                    "scope":
                        "Production Models, Store, sessions and service; drawing-only adapter. GUI and iPhone acceptance are separate.",
                ], to: output, name: "native.json")
        } catch {
            await withTaskCancellationShield {
                if let ticket = library.state.installations[artifact.id]?.requestID {
                    library.send(.cancelRequested(artifact.id))
                    await assembly.runner.run(.stopTransfer(artifactID: artifact.id, ticket), send: { _ in })
                }
                if let session = library.state.session {
                    await assembly.runner.run(.stopDemo(session, session.id), send: { _ in })
                }
                await assembly.runner.release(owner: library.state.owner)
            }
            throw error
        }
    }

    private func predict(
        library: Store<ModelLibraryFeature>, assembly: ModelLibraryAssembly, adapter: ChronosNativeAdapter,
        completed: inout AsyncStream<ModelLibraryEffect>.Iterator, installed: ValidatedModelResource,
        reference: ChronosScenesReference, storage: ManagedModelStore
    ) async throws -> [[String: Any]] {
        let rendering = ModelLibraryRendering(registry: assembly.registry, runner: assembly.runner, library: library)
        library.send(.openDemo(ChronosDemoCatalog.selection))
        let session = try #require(library.state.session)
        _ = rendering.content(for: session)
        library.send(.requestDemoPreparation(session.id))
        while let effect = await completed.next(isolation: #isolation) { if case .prepareDemo = effect { break } }
        try #require(library.state.demoPreparation == .available)
        _ = rendering.content(for: session)
        let store = try #require(adapter.store)
        var actions = adapter.actions.stream.makeAsyncIterator()
        while await actions.next(isolation: #isolation) != nil {
            if store.state.phase == .ready || store.state.error != nil { break }
        }
        try #require(
            store.state.phase == .ready && store.state.error == nil && store.state.mode == .live
                && store.state.forecast == nil)
        try #require(await storage.activeLeases(identity: installed.identity) == 1)
        let catalog = try #require(store.state.catalog)
        var rows: [[String: Any]] = []
        for scenario in catalog.scenarios {
            store.send(.selectScenario(scenario.id))
            store.send(.predict)
            while await actions.next(isolation: #isolation) != nil {
                if store.state.phase == .ready || store.state.error != nil { break }
            }
            let forecast = try #require(store.state.forecast)
            try #require(
                store.state.phase == .ready && store.state.error == nil && forecast.mode == .live
                    && forecast.scenarioID == scenario.id && forecast.originStep == 0)
            try #require(forecast.identity?.manifest.sha256 == installed.revision)
            for area in catalog.areas {
                let expected = try #require(
                    reference.cases.first {
                        $0.scenarioID == scenario.id && $0.areaID == area.id && $0.originStep == 0
                    })
                let raw = try #require(forecast.area(area.id)?.raw)
                try #require(expected.quantiles.count == 1 && expected.quantiles[0].count == 21)
                try #require(raw.quantiles.count == 21)
                for row in 0..<21 {
                    try #require(expected.quantiles[0][row].count == 16 && raw.quantiles[row].count == 16)
                    for step in 0..<16 {
                        let value = expected.quantiles[0][row][step]
                        try #require(
                            value.isFinite
                                && abs(Double(raw.quantiles[row][step]) - value) <= 0.0001 + 0.0001 * abs(value))
                    }
                }
                rows.append(["scenarioID": scenario.id, "areaID": area.id, "quantiles": raw.quantiles])
            }
        }
        try #require(rows.count == 60)
        library.send(.closeDemo)
        while let effect = await completed.next(isolation: #isolation) { if case .stopDemo = effect { break } }
        try #require(await storage.activeLeases(identity: installed.identity) == 0)
        try #require(adapter.store?.state.ended == true && library.state.session == nil)
        await assembly.runner.release(owner: library.state.owner)
        return rows
    }

    private func newOutput(_ path: String) throws -> URL {
        let input = URL(filePath: path)
        let pointer = try #require(realpath(input.deletingLastPathComponent().path(percentEncoded: false), nil))
        defer { free(pointer) }
        let output = URL(filePath: String(cString: pointer)).appending(path: input.lastPathComponent)
        try #require(!FileManager.default.fileExists(atPath: output.path))
        return output
    }

    private func fileRows(_ files: [ModelArtifactFile]) -> [[String: Any]] {
        files.map { ["path": $0.path, "bytes": $0.bytes, "sha256": $0.sha256] }
    }

    private func write(_ value: [String: Any], to output: URL, name: String) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appending(path: name), options: .withoutOverwriting)
    }
}

@concurrent
private func inventory(_ directory: URL, files: [ModelArtifactFile]) async throws -> [ModelArtifactFile] {
    try files.map { file in
        try Task.checkCancellation()
        let path = directory.appending(path: file.path)
        let size = try #require(
            (try FileManager.default.attributesOfItem(atPath: path.path)[.size] as? NSNumber)?.int64Value)
        let handle = try FileHandle(forReadingFrom: path)
        defer { try? handle.close() }
        var hash = SHA256()
        while try autoreleasepool(invoking: {
            guard let data = try handle.read(upToCount: 1_048_576), !data.isEmpty else { return false }
            try Task.checkCancellation()
            hash.update(data: data)
            return true
        }) {}
        return ModelArtifactFile(
            path: file.path, bytes: size, sha256: hash.finalize().map { String(format: "%02x", $0) }.joined())
    }
}
