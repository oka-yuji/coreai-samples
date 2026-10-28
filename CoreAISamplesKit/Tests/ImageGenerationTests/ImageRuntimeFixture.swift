import Foundation
import ImageGenerationCore
import ModelLibraryTestSupport
import Synchronization
import Testing

@testable import ImageGenerationRuntime
@testable import ModelLibraryFeature

struct ImageRuntimeFixture: Sendable {
    let bundle: ImageBundleFixture
    let resource: ValidatedModelResource
    let budget: ImageMemoryBudget

    init(operation: String = "generate-and-edit", assetFolder: String = "", headersBesideAssets: Bool = true) throws {
        bundle = try ImageBundleFixture { manifest, files in
            files["config/processor/tokenizer_config.json"] = Data("{}".utf8)
            files["config/vae/config.json"] = try JSONSerialization.data(withJSONObject: [
                "latents_mean": [Float](repeating: 0, count: 64), "latents_std": [Float](repeating: 1, count: 64),
            ])
            try Self.arrange(
                &manifest, &files, operation: operation, assetFolder: assetFolder,
                headersBesideAssets: headersBesideAssets)
        }
        resource = ValidatedModelResource(
            modelID: bundle.model.id, artifactID: bundle.artifact.id, contract: bundle.artifact.contract,
            revision: try #require(bundle.artifact.sourceRevision),
            upstreamRevision: bundle.artifact.upstream?.revision,
            directory: bundle.source, sizeBytes: bundle.files.values.reduce(0) { $0 + Int64($1.count) },
            sourceKind: .localManifest)
        budget = try ImageMemoryBudget(
            profileID: "fixture-qmi-profile", manifestSHA256: try #require(bundle.artifact.sourceRevision),
            components: Dictionary(
                uniqueKeysWithValues: ImageModelComponent.allCases.map {
                    (
                        $0,
                        try ImageComponentMemoryBudget(
                            loadingAdditionalBytes: 1_048_576, inferenceAdditionalBytes: 524_288)
                    )
                }))
    }

    func remove() { bundle.remove() }

    private static func arrange(
        _ manifest: inout [String: Any], _ files: inout [String: Data], operation: String, assetFolder: String,
        headersBesideAssets: Bool
    ) throws {
        var profile = try #require(manifest["profile"] as? [String: Any])
        var workload = try #require(profile["workload"] as? [String: Any])
        guard workload["operation"] as? String != operation || !assetFolder.isEmpty || !headersBesideAssets else {
            return
        }
        workload["operation"] = operation
        profile["workload"] = workload
        var components = try #require(profile["components"] as? [String: [String: Any]])
        for (kind, component) in components {
            let asset = try #require(component["asset"] as? String)
            let moved = assetFolder.isEmpty ? asset : assetFolder + "/" + asset
            let weights = files.removeValue(forKey: asset + "/weights.bin")
            files[moved + "/weights.bin"] = try #require(weights)
            components[kind]?["asset"] = moved
        }
        profile["components"] = components
        manifest["profile"] = profile
        for (kind, component) in components {
            let asset = try #require(component["asset"] as? String)
            let header: [String: Any] = [
                "format": 2, "kind": kind, "asset": URL(filePath: asset).lastPathComponent,
                "conversion_complete": true, "state_names": [], "profile": profile,
                "inputs": try #require(component["inputs"]), "outputs": try #require(component["outputs"]),
            ]
            files.removeValue(forKey: kind + "/component.json")
            let folder = headersBesideAssets ? asset.split(separator: "/").dropLast().joined(separator: "/") : kind
            files[folder + "/component.json"] = try JSONSerialization.data(withJSONObject: header, options: .sortedKeys)
        }
    }

    func pipeline(
        backend: any ImageComponentBackend, memory: ImageTestMemory = ImageTestMemory(),
        journal: (any ImageCheckpointSink)? = nil,
        permission: @escaping ImageRuntimePipeline.Permission = { true }
    ) -> ImageRuntimePipeline {
        ImageRuntimePipeline(
            resource: resource, budget: budget, backend: backend, permission: permission, memory: memory.read,
            journal: journal,
            preparePrompt: { _, _, reference in
                try ImagePromptLayout(
                    tokens: [1] + (reference ? Array(repeating: ImagePromptLayout.imageToken, count: 256) : []) + [2],
                    drop: 1, hasReference: reference)
            })
    }

    func request(editing: Bool = false) throws -> ImageGenerationRequest {
        let image =
            editing
            ? try ImageGenerationPixels(width: 512, height: 512, rgba: [UInt8](repeating: 128, count: 512 * 512 * 4))
            : nil
        return ImageGenerationRequest(
            id: UUID(), prompt: "fixture", seed: 1, reference: image,
            initialNoise: [Float](repeating: 0.5, count: 65_536))
    }

    func session(owner: UUID = UUID(), number: Int = 1) -> ModelDemoSession {
        ModelDemoSession(
            id: LibraryTicket(owner: owner, number: number),
            selection: DemoSelection(modelID: resource.modelID, demoID: "image-generation"),
            resource: resource)
    }
}

final class ImageTestMemory: Sendable {
    private let available = Mutex(UInt64(8_000_000_000))

    func setAvailable(_ value: UInt64) { available.withLock { $0 = value } }

    func read() -> ImageMemoryObservation {
        ImageMemoryObservation(
            scope: .processDirtyBudget, availableBytes: available.withLock { $0 }, footprintBytes: 100_000_000,
            physicalMemoryBytes: 16_000_000_000)
    }
}

enum ImageTestError: Error, Equatable { case native, unload, missingEvent }

actor ImageFakeBackend: ImageComponentBackend {
    let events: AsyncStream<String>
    private let sink: AsyncStream<String>.Continuation
    private var held: (String, AsyncStream<Void>)?
    private var loaded: ImageGenerationProfile.Component?
    private var calls = 0
    private(set) var history: [String] = []
    private var failures: Set<String> = []
    private var nonfinite = false

    init() {
        (events, sink) = AsyncStream.makeStream()
    }

    func hold(_ event: String, until gate: AsyncStream<Void>) { held = (event, gate) }
    func fail(_ event: String) { failures.insert(event) }
    func clearFailures() { failures.removeAll() }
    func returnNonfinite() { nonfinite = true }
    nonisolated func finishEvents() { sink.finish() }

    func load(directory: URL, component: ImageGenerationProfile.Component) async throws -> ImageComponentLoadMetrics {
        #expect(loaded == nil)
        loaded = component
        calls = 0
        let event = "load:" + component.asset.split(separator: "/")[0]
        try await signal(String(event))
        return ImageComponentLoadMetrics(
            cacheHit: true, cacheLookupSeconds: 0, specializationSeconds: nil, functionLoadSeconds: 0)
    }

    func call(_ values: [String: ImageTensorData]) async throws -> [String: ImageTensorData] {
        let description = try #require(loaded)
        let kind = String(description.asset.split(separator: "/")[0])
        try await signal("call:\(kind):\(calls)")
        calls += 1
        return try Dictionary(
            uniqueKeysWithValues: description.outputs.map {
                ($0.name, .floats([Float](repeating: nonfinite ? .nan : 0, count: try $0.elementCount())))
            })
    }

    func unload() async throws {
        let kind = loaded.map { String($0.asset.split(separator: "/")[0]) } ?? "none"
        try await signal("unload:" + kind)
        loaded = nil
    }

    private func signal(_ event: String) async throws {
        history.append(event)
        sink.yield(event)
        if let (name, gate) = held, name == event {
            held = nil
            for await _ in gate { break }
        }
        if failures.contains(event) { throw ImageTestError.native }
    }
}

func imageEvent(_ expected: String, from iterator: inout AsyncStream<String>.Iterator) async throws {
    while let event = await iterator.next(isolation: #isolation) {
        if event == expected { return }
    }
    throw ImageTestError.missingEvent
}

actor ImageFakeTrace: ImageTraceSink {
    private(set) var names: [String] = []
    func record(name: String, values: [String: ImageTensorData], contracts: [ImageTensorContract]) {
        names.append(name)
    }
}
