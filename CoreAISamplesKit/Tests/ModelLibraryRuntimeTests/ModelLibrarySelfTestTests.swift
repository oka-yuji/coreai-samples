import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import Synchronization
import Testing

@testable import ModelLibraryRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ModelLibrarySelfTestTests {
    @Test(
        "CD-03 A downloadable manifest chooses download without a folder and reports a simulated authentication failure"
    )
    func privateDownloadFailure() async throws {
        let artifact = try ChronosModelArtifact.make()
        let catalog = try catalog(artifact)
        let folder = URL.temporaryDirectory.appending(path: "manifest-selftest-\(UUID())")
        defer { try? FileManager.default.removeItem(at: folder) }
        let requests = Mutex<[URLRequest]>([])
        let transfer = ArtifactTransfer(
            fileLoader: { request, _ in
                requests.withLock { $0.append(request) }
                let url = try #require(request.url)
                return try #require(
                    HTTPURLResponse(url: url, statusCode: 401, httpVersion: nil, headerFields: nil))
            }, availableCapacity: { _ in .max })
        let assembly = ModelLibraryAssembly(
            adapters: [], store: ManagedModelStore(root: folder, availableCapacity: { _ in .max }), transfer: transfer)
        var connected = false
        let report = try await ModelLibrarySelfTest.check(
            catalog: catalog, root: folder, registry: nil, capabilities: nil, acquire: artifact.id, assembly: assembly,
            afterAcquisition: .init { _ in
                connected = true
                return []
            })
        let result = try #require(report.acquisition?.artifacts.first)
        #expect(result.route == "download" && result.state == "failed" && result.inspection == "missing")
        #expect(result.revision == ChronosModelArtifact.manifestSHA256 && result.identity == nil)
        #expect(result.failure?.contains("HTTP 401") == true)
        #expect(report.failures.count == 2 && !connected)
        #expect(report.failures.last == "Additional checks require successful acquisition")
        let sent = requests.withLock { $0 }
        #expect(!sent.isEmpty)
        #expect(
            sent.allSatisfy {
                $0.url?.absoluteString.hasPrefix(
                    "https://huggingface.co/okayuji/chronos-2-coreai-fp32/resolve/d3ead488046498faa8b3b5103b1923e52a313d0e/bundle/"
                ) == true
                    && ($0.allHTTPHeaderFields ?? [:]).isEmpty
            })
    }

    @Test("CD-03 Selecting an import folder takes priority over the locator and a rejected import stays a failure")
    func explicitImport() async throws {
        let artifact = try ChronosModelArtifact.make()
        let catalog = try catalog(artifact)
        let base = URL.temporaryDirectory.appending(path: "manifest-import-selftest-\(UUID())")
        defer { try? FileManager.default.removeItem(at: base) }
        let root = base.appending(path: "library")
        let source = base.appending(path: "selected")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let transfer = ArtifactTransfer(fileLoader: { _, _ in
            Issue.record("An explicit import must not make a download request")
            throw URLError(.badURL)
        })
        let assembly = ModelLibraryAssembly(
            adapters: [], store: ManagedModelStore(root: root, availableCapacity: { _ in .max }), transfer: transfer)
        let report = try await ModelLibrarySelfTest.check(
            catalog: catalog, root: root, registry: nil, capabilities: nil,
            acquire: artifact.id, assembly: assembly, importBundle: source)
        let result = try #require(report.acquisition?.artifacts.first)
        #expect(result.route == "local-import" && result.state == "failed" && result.inspection == "missing")
        #expect(result.failure != nil && result.identity == nil && report.failures.count == 1)
        #expect(try FileManager.default.contentsOfDirectory(atPath: source.path).isEmpty)
    }

    @Test("CD-03 An uninstalled local-only artifact still requires a selected import folder")
    func localOnlyRequiresFolder() async throws {
        let artifact = try IrodoriModelArtifact.make()
        let root = URL.temporaryDirectory.appending(path: "local-selftest-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let catalog = try catalog(artifact)
        let error = await #expect(throws: (any Error).self) {
            try await ModelLibrarySelfTest.check(
                catalog: catalog, root: root, registry: nil, capabilities: nil, acquire: artifact.id)
        }
        #expect(String(describing: error).contains("A selected folder is required"))
    }

    private func catalog(_ artifact: ModelArtifact) throws -> ModelCatalog {
        try ModelCatalog(
            models: [
                LibraryModel(
                    id: "fixture", name: "Fixture", category: "Fixture", artifacts: [artifact],
                    defaultArtifactID: artifact.id, platforms: [.macOS, .iOS],
                    demos: [ModelDemoLink(demoID: "preview", platforms: [.macOS, .iOS])])
            ],
            demos: [
                DemoDescriptor(
                    id: "preview", title: "Preview", systemImage: "chart.line.uptrend.xyaxis", summary: "Preview",
                    platforms: [.macOS, .iOS])
            ])
    }
}
