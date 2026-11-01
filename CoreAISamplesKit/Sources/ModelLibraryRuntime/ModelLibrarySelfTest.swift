import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer

public enum ModelLibrarySelfTest {
    public struct AcquisitionCheck {
        let run: @MainActor (ModelLibraryAssembly.Library) async -> [String]

        public init(_ run: @escaping @MainActor (ModelLibraryAssembly.Library) async -> [String]) {
            self.run = run
        }
    }

    struct Report: Encodable {
        struct Registry: Encodable {
            let connectedDemos: [String]
            let unconnectedDemos: [String]
            let matchesCatalog: Bool
        }

        struct Root: Encodable {
            let path: String
            let exists: Bool
            let excludedFromBackup: Bool
        }

        struct Artifact: Encodable {
            let modelID: String
            let artifactID: String
            let contract: String
            let distribution: String?
            let sourceKind: ModelSourceKind?
            let localManifestSHA256: String?
            let upstreamRevision: String?
            let files: Int
            let inspection: String
        }

        struct Acquisition: Encodable {
            struct Identity: Encodable {
                let modelID: String
                let artifactID: String
                let contract: String
                let revision: String
                let upstreamRevision: String?
                let sourceKind: ModelSourceKind
            }

            struct Result: Encodable {
                let modelID: String
                let artifactID: String
                let revision: String?
                let route: String
                let definedBytes: Int64
                let measuredBytes: Int64?
                let seconds: Double?
                let state: String
                let inspection: String
                let identity: Identity?
                let failure: String?
            }

            let target: String
            let refreshSeconds: Double?
            let artifacts: [Result]
        }

        let platform: String
        let scope: String
        let catalogValidated: Bool
        let catalogDemos: [String]
        let visibleModels: Int
        let visibleDemos: Int
        let registry: Registry?
        let capabilities: [String: String]?
        let store: Root
        let artifacts: [Artifact]
        let acquisition: Acquisition?
        let failures: [String]
    }

    private struct Failure: Error, CustomStringConvertible {
        let description: String
    }

    public static func run(
        output: URL, root: URL, registry: ModelDemoAdapterRegistry? = nil,
        capabilities: [DemoCapability: CapabilityState]? = nil, acquire target: String? = nil,
        assembly: ModelLibraryAssembly? = nil, importBundle: URL? = nil, unconnectedDemoIDs: Set<String> = [],
        afterAcquisition: AcquisitionCheck? = nil,
        catalog: @MainActor () throws -> ModelCatalog = { try IOSDemoCatalog.make() }
    ) async -> Int32 {
        let failures: [String]
        do {
            let report: any Encodable
            do {
                let checked = try await check(
                    catalog: catalog(), root: root,
                    registry: registry, capabilities: capabilities,
                    acquire: target, assembly: assembly, importBundle: importBundle,
                    unconnectedDemoIDs: unconnectedDemoIDs, afterAcquisition: afterAcquisition)
                failures = checked.failures
                report = checked
            } catch {
                failures = ["\(error)"]
                report = ["failures": failures]
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(report).write(to: output, options: .atomic)
        } catch {
            FileHandle.standardError.write(Data("MODEL LIBRARY SELFTEST could not write the report: \(error)\n".utf8))
            return 1
        }
        for failure in failures {
            FileHandle.standardError.write(Data("[model-library-selftest] \(failure)\n".utf8))
        }
        let verdict = failures.isEmpty ? "PASS" : "FAIL"
        FileHandle.standardOutput.write(Data("MODEL LIBRARY SELFTEST \(verdict) \(output.relativePath)\n".utf8))
        return failures.isEmpty ? 0 : 1
    }

    static func check(
        catalog: ModelCatalog, root: URL, registry: ModelDemoAdapterRegistry?,
        capabilities: [DemoCapability: CapabilityState]?, acquire target: String? = nil,
        assembly: ModelLibraryAssembly? = nil, importBundle: URL? = nil, unconnectedDemoIDs: Set<String> = [],
        afterAcquisition: AcquisitionCheck? = nil
    ) async throws -> Report {
        let demos = catalog.demos.map(\.id).sorted()
        let connected = registry?.connectedDemos.sorted()
        var failures: [String] = []
        var connectionsMatch = true
        if let connected {
            do {
                try catalog.validateConnections(adapterIDs: connected, unconnectedDemoIDs: unconnectedDemoIDs)
            } catch {
                connectionsMatch = false
                failures.append("registry: \(error)")
            }
        }
        guard importBundle == nil || target != nil else {
            throw Failure(description: "Local import requires an exact artifact target")
        }
        guard afterAcquisition == nil || target != nil else {
            throw Failure(description: "Additional acquisition checks require an artifact target")
        }
        let artifacts = try await inspect(catalog, root: root)
        for artifact in artifacts where artifact.sourceKind == nil && artifact.inspection != "missing" {
            failures.append(
                "\(artifact.artifactID): an artifact without a distribution was inspected as \(artifact.inspection)")
        }
        var acquisition: Report.Acquisition?
        if let target {
            let (acquired, additionalFailures) = try await acquire(
                target, catalog: catalog, root: root, capabilities: capabilities ?? [:],
                assembly: assembly
                    ?? ModelLibraryAssembly(
                        adapters: [], store: ManagedModelStore(root: root), transfer: ArtifactTransfer()),
                installed: Set(artifacts.filter { $0.inspection == "ready" }.map(\.artifactID)),
                importBundle: importBundle, afterAcquisition: afterAcquisition)
            if acquired.artifacts.isEmpty { failures.append("acquire: \(target) names no catalog artifact") }
            failures += acquired.artifacts.compactMap(problem)
            failures += additionalFailures
            acquisition = acquired
        }
        let sections = catalog.sections(platform: .iOS, audience: .release)
        #if os(iOS)
            let platform = "iOS"
        #else
            let platform = "macOS"
        #endif
        let scope =
            registry == nil
            ? "Catalog and store checks only. The demo adapters and capability states belong to the iOS app."
            : "Catalog, registry, capability and store checks of the iOS app."
        return Report(
            platform: platform,
            scope: scope + (target.map { " Acquisition of \($0) through the library store." } ?? ""),
            catalogValidated: true, catalogDemos: demos, visibleModels: sections.count,
            visibleDemos: sections.reduce(0) { $0 + $1.demos.count },
            registry: connected.map {
                Report.Registry(
                    connectedDemos: $0, unconnectedDemos: unconnectedDemoIDs.sorted(), matchesCatalog: connectionsMatch)
            },
            capabilities: capabilities.map { states in
                Dictionary(uniqueKeysWithValues: states.map { ($0.key.rawValue, "\($0.value)") })
            },
            store: Report.Root(
                path: displayPath(root),
                exists: FileManager.default.fileExists(atPath: root.path(percentEncoded: false)),
                excludedFromBackup: (try? root.resourceValues(forKeys: [.isExcludedFromBackupKey]))?
                    .isExcludedFromBackup == true),
            artifacts: artifacts, acquisition: acquisition, failures: failures)
    }

    private static func inspect(_ catalog: ModelCatalog, root: URL) async throws -> [Report.Artifact] {
        let store = ManagedModelStore(root: root)
        var artifacts: [Report.Artifact] = []
        for model in catalog.models {
            for artifact in model.artifacts {
                let inspection = try await store.inspect(model: model, artifact: artifact)
                artifacts.append(
                    Report.Artifact(
                        modelID: model.id, artifactID: artifact.id, contract: artifact.contract,
                        distribution: artifact.download.map { "\($0.repository)@\($0.revision)" },
                        sourceKind: artifact.sourceKind, localManifestSHA256: artifact.localBundle?.manifestSHA256,
                        upstreamRevision: artifact.upstream?.revision, files: artifact.files.count,
                        inspection: describe(inspection)))
            }
        }
        return artifacts
    }

    private static func acquire(
        _ target: String, catalog: ModelCatalog, root: URL, capabilities: [DemoCapability: CapabilityState],
        assembly: ModelLibraryAssembly, installed: Set<String>, importBundle: URL?,
        afterAcquisition: AcquisitionCheck?
    ) async throws -> (Report.Acquisition, [String]) {
        let selected = catalog.models.flatMap { model in model.artifacts.map { (model, $0) } }
            .filter { target == "all" || $0.1.id == target }
        guard !selected.isEmpty else {
            return (Report.Acquisition(target: target, refreshSeconds: nil, artifacts: []), [])
        }
        if importBundle != nil {
            guard selected.count == 1, selected[0].1.id == target, selected[0].1.sourceKind == .localManifest else {
                throw Failure(description: "Local import requires one exact local artifact target")
            }
        }
        guard
            importBundle != nil
                || selected.allSatisfy({ $0.1.sourceKind != .localManifest || installed.contains($0.1.id) })
        else {
            throw Failure(description: "A selected folder is required before acquiring an uninstalled local artifact")
        }
        let context = assembly.makeLibrary(catalog: catalog, capabilities: capabilities, owner: UUID())
        let library = context.store
        let clock = ContinuousClock()
        let refresh = clock.now
        library.send(.refresh)
        guard await settle(library, catalog.models.flatMap(\.artifacts).map(\.id)) else {
            throw Failure(description: "acquire: the library inspection did not finish within 30 minutes")
        }
        let refreshSeconds = seconds(refresh.duration(to: clock.now))
        let store = ManagedModelStore(root: root)
        var results: [Report.Acquisition.Result] = []
        for (model, artifact) in selected {
            var route = installed.contains(artifact.id) ? "installed" : "migration"
            var elapsed: Double?
            var timedOut = false
            if library.state.installations[artifact.id]?.phase != .ready || importBundle != nil {
                route = artifact.sourceKind == .localManifest ? "local-import" : "download"
                let start = clock.now
                if artifact.sourceKind == .localManifest {
                    guard let importBundle else {
                        throw Failure(description: "A selected folder is required for local import")
                    }
                    library.send(.importRequested(artifact.id, importBundle))
                } else {
                    library.send(.downloadRequested(artifact.id))
                }
                if await !settle(library, [artifact.id]) {
                    timedOut = true
                    library.send(.cancelRequested(artifact.id))
                    _ = await settle(library, [artifact.id])
                }
                elapsed = seconds(start.duration(to: clock.now))
            }
            let installation = library.state.installations[artifact.id] ?? ModelInstallation()
            let resource = installation.resource
            let inspection = try await store.inspect(model: model, artifact: artifact)
            results.append(
                Report.Acquisition.Result(
                    modelID: model.id, artifactID: artifact.id, revision: artifact.sourceRevision, route: route,
                    definedBytes: artifact.totalFileBytes ?? 0,
                    measuredBytes: resource.map { size(of: $0.directory) }, seconds: elapsed,
                    state: timedOut ? "timed out" : "\(installation.phase)", inspection: describe(inspection),
                    identity: resource.map {
                        Report.Acquisition.Identity(
                            modelID: $0.modelID, artifactID: $0.artifactID, contract: $0.contract,
                            revision: $0.revision, upstreamRevision: $0.upstreamRevision, sourceKind: $0.sourceKind)
                    },
                    failure: timedOut
                        ? "The acquisition did not finish within 30 minutes" : installation.failure.map { "\($0)" }))
        }
        let additionalFailures: [String]
        if let afterAcquisition {
            additionalFailures =
                results.compactMap(problem).isEmpty
                ? await afterAcquisition.run(context) : ["Additional checks require successful acquisition"]
        } else {
            additionalFailures = []
        }
        return (
            Report.Acquisition(target: target, refreshSeconds: refreshSeconds, artifacts: results), additionalFailures
        )
    }

    private static func settle(_ library: Store<ModelLibraryFeature>, _ artifactIDs: [String]) async -> Bool {
        await withTaskGroup(of: Bool.self) { group in
            group.addTask { await idle(library, artifactIDs) }
            group.addTask {
                try? await Task.sleep(for: .seconds(1800))
                return false
            }
            let settled = await group.next() ?? false
            group.cancelAll()
            return settled
        }
    }

    private static func idle(_ library: Store<ModelLibraryFeature>, _ artifactIDs: [String]) async -> Bool {
        for await state in Observations({ library.state })
        where !artifactIDs.contains(where: { state.installations[$0]?.phase.isBusy == true }) {
            return true
        }
        return false
    }

    private static func problem(_ result: Report.Acquisition.Result) -> String? {
        guard
            result.failure != nil || result.state != "ready" || result.inspection != "ready"
                || result.measuredBytes != result.definedBytes
        else { return nil }
        return "\(result.artifactID): acquisition \(result.state), inspection \(result.inspection)"
            + (result.failure.map { ", \($0)" } ?? "")
    }

    private static func describe(_ inspection: ManagedInspection) -> String {
        switch inspection {
        case .ready: "ready"
        case .missing: "missing"
        case .invalid(let error): "invalid: \(error)"
        }
    }

    private static func size(of directory: URL) -> Int64 {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey]
        guard let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: Array(keys))
        else { return 0 }
        var total: Int64 = 0
        for case let file as URL in files {
            guard let values = try? file.resourceValues(forKeys: keys), values.isRegularFile == true else { continue }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }

    private static func seconds(_ duration: Duration) -> Double {
        let (whole, fraction) = duration.components
        return ((Double(whole) + Double(fraction) / 1e18) * 10).rounded() / 10
    }

    private static func displayPath(_ url: URL) -> String {
        let home = URL.homeDirectory.standardizedFileURL.pathComponents
        let components = url.standardizedFileURL.pathComponents
        guard components.starts(with: home) else { return url.path(percentEncoded: false) }
        return (["~"] + components.dropFirst(home.count)).joined(separator: "/")
    }
}
