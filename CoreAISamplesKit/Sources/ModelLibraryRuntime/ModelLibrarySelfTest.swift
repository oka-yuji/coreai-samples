import Foundation
import ModelLibraryFeature
import ModelLibraryStorage

public enum ModelLibrarySelfTest {
    private struct Report: Encodable {
        struct Registry: Encodable {
            let connectedDemos: [String]
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
            let upstreamRevision: String?
            let files: Int
            let inspection: String
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
        let failures: [String]
    }

    public static func run(
        output: URL, root: URL, registry: ModelDemoAdapterRegistry? = nil,
        capabilities: [DemoCapability: CapabilityState]? = nil
    ) async -> Int32 {
        let failures: [String]
        do {
            let report: any Encodable
            do {
                let checked = try await check(root: root, registry: registry, capabilities: capabilities)
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

    private static func check(
        root: URL, registry: ModelDemoAdapterRegistry?, capabilities: [DemoCapability: CapabilityState]?
    ) async throws -> Report {
        let catalog = try IOSDemoCatalog.make()
        let demos = catalog.demos.map(\.id).sorted()
        let connected = registry?.connectedDemos.sorted()
        var failures: [String] = []
        if let connected, connected != demos {
            failures.append("registry: connected demos \(connected) differ from catalog demos \(demos)")
        }
        let artifacts = try await inspect(catalog, root: root)
        for artifact in artifacts where artifact.distribution == nil && artifact.inspection != "missing" {
            failures.append(
                "\(artifact.artifactID): an artifact without a distribution was inspected as \(artifact.inspection)")
        }
        let sections = catalog.sections(platform: .iOS, audience: .release)
        #if os(iOS)
            let platform = "iOS"
        #else
            let platform = "macOS"
        #endif
        return Report(
            platform: platform,
            scope: registry == nil
                ? "Catalog and store checks only. The demo adapters and capability states belong to the iOS app."
                : "Catalog, registry, capability and store checks of the iOS app.",
            catalogValidated: true, catalogDemos: demos, visibleModels: sections.count,
            visibleDemos: sections.reduce(0) { $0 + $1.demos.count },
            registry: connected.map { Report.Registry(connectedDemos: $0, matchesCatalog: $0 == demos) },
            capabilities: capabilities.map { states in
                Dictionary(uniqueKeysWithValues: states.map { ($0.key.rawValue, "\($0.value)") })
            },
            store: Report.Root(
                path: displayPath(root),
                exists: FileManager.default.fileExists(atPath: root.path(percentEncoded: false)),
                excludedFromBackup: (try? root.resourceValues(forKeys: [.isExcludedFromBackupKey]))?
                    .isExcludedFromBackup == true),
            artifacts: artifacts, failures: failures)
    }

    private static func inspect(_ catalog: ModelCatalog, root: URL) async throws -> [Report.Artifact] {
        let store = ManagedModelStore(root: root)
        var artifacts: [Report.Artifact] = []
        for model in catalog.models {
            for artifact in model.artifacts {
                let inspection =
                    switch try await store.inspect(model: model, artifact: artifact) {
                    case .ready: "ready"
                    case .missing: "missing"
                    case .invalid(let error): "invalid: \(error)"
                    }
                artifacts.append(
                    Report.Artifact(
                        modelID: model.id, artifactID: artifact.id, contract: artifact.contract,
                        distribution: artifact.download.map { "\($0.repository)@\($0.revision)" },
                        upstreamRevision: artifact.upstream?.revision, files: artifact.files.count,
                        inspection: inspection))
            }
        }
        return artifacts
    }

    private static func displayPath(_ url: URL) -> String {
        let home = URL.homeDirectory.standardizedFileURL.pathComponents
        let components = url.standardizedFileURL.pathComponents
        guard components.starts(with: home) else { return url.path(percentEncoded: false) }
        return (["~"] + components.dropFirst(home.count)).joined(separator: "/")
    }
}
