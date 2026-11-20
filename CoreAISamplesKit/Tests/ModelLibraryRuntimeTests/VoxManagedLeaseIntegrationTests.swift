import Foundation
import Testing

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct VoxManagedLeaseIntegrationTests {
    private var manifests: URL {
        URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "ModelLibraryStorageTests/Resources", directoryHint: .isDirectory)
    }
    @Test(
        "VCL-02 Leaving a Mac speech view keeps the file lease until uncooperative work and physical shutdown finish",
        arguments: [true, false])
    func screenExit(duringOpen: Bool) async throws {
        try await VoxManagedLeaseScenarios.closing(manifests: manifests, duringOpen: duringOpen)
    }
    @Test(
        "VCL-01 A cold generation refuses a replacement at the same managed path until the new registration is opened")
    func samePath() async throws {
        try await VoxManagedLeaseScenarios.replacement(manifests: manifests)
    }
    @Test(
        "VCL-03 The runner releases only the abandoning window's display usage for equal and distinct variants",
        arguments: [true, false])
    func twoWindows(same: Bool) async throws {
        try await VoxManagedLeaseScenarios.windows(manifests: manifests, same: same)
    }
    @Test("VCL-04 A returned physical lease survives its view store and a retired acquirer cannot grant another")
    func actualWeakOwnership() async throws {
        try await VoxManagedLeaseScenarios.lostWindow(manifests: manifests)
    }
}
