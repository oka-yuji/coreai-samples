import Testing

@testable import CoreAIBackend

@Suite struct ProcessFootprintTests {
    @Test("The graphics footprint of the current process can be read")
    func readsGraphicsFootprint() throws {
        _ = try ProcessFootprint.graphicsBytes()
    }
}
