import ChronosCore
import Testing

@testable import ChronosRuntime

struct ChronosManagedIdentityTests {
    @Test("The native loader rejects a managed manifest mismatch before attempting model construction")
    func mismatchedSnapshot() async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        try folder.writeBundle()
        await #expect(throws: ChronosError.invalidBundle("managed manifest identity")) {
            _ = try await ChronosNativePredictor.load(bundleURL: folder.url, expectedManifestSHA256: "wrong")
        }
    }
}
