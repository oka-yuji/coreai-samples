import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import Synchronization
import Testing

@testable import ModelLibraryRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct MobileImageLabCatalogTests {
    @Test("QD-04 A synthetic download failure keeps the canonical public inventory uninstalled and skips connection")
    func failedDownload() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let requests = Mutex<[URL]>([])
        let transfer = ArtifactTransfer(
            fileLoader: { request, _ in
                if let url = request.url { requests.withLock { $0.append(url) } }
                throw URLError(.notConnectedToInternet)
            }, availableCapacity: { _ in .max })
        let catalog = try IOSDemoCatalog.make()
        let assembly = ModelLibraryAssembly(
            adapters: [], store: ManagedModelStore(root: root, availableCapacity: { _ in .max }), transfer: transfer)
        var connectionCalls = 0
        let result = try await ModelLibrarySelfTest.check(
            catalog: catalog, root: root, registry: assembly.registry, capabilities: [:],
            acquire: MobileImageLabArtifact.published.id, assembly: assembly,
            unconnectedDemoIDs: Set(catalog.demos.map(\.id)),
            afterAcquisition: .init { _ in
                connectionCalls += 1
                return []
            })
        let urls = requests.withLock { $0 }
        #expect(!urls.isEmpty)
        #expect(
            urls.allSatisfy {
                $0.absoluteString.hasPrefix(
                    "https://huggingface.co/okayuji/mobile-image-lab-coreai/resolve/fdd2b8cb17f48976207039f1f780360ba60275dc/bundle/"
                )
            })
        let acquisition = try #require(result.acquisition?.artifacts.first)
        #expect(acquisition.artifactID == MobileImageLabArtifact.published.id)
        #expect(acquisition.state != "ready" && acquisition.failure != nil)
        #expect(!result.failures.isEmpty && connectionCalls == 0)
    }
}
