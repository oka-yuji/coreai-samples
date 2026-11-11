import Foundation
import Testing

@testable import ModelLibraryFeature

struct ChronosPreviewPolicyTests {
    @Test("Chronos explicitly enables its asset-free preview while other demos retain disabled controls")
    func optIn() {
        let chronos = ChronosDemoCatalog.demo(platforms: [.iOS], searchTerms: [:])
        let other = DemoDescriptor(
            id: "other", title: "Other", systemImage: "circle", summary: "Other", platforms: [.iOS])
        #expect(chronos.allowsPreviewInteraction)
        #expect(!other.allowsPreviewInteraction)
    }

    @Test("Older demo descriptors keep previews disabled when the optional interaction flag is absent")
    func compatibility() throws {
        let old = DemoDescriptor(
            id: "other", title: "Other", systemImage: "circle", summary: "Other", platforms: [.iOS])
        let oldData = try JSONEncoder().encode(old)
        let object = try #require(JSONSerialization.jsonObject(with: oldData) as? [String: Any])
        #expect(object["interactivePreview"] == nil)
        #expect(try !JSONDecoder().decode(DemoDescriptor.self, from: oldData).allowsPreviewInteraction)
        let interactive = DemoDescriptor(
            id: "preview", title: "Preview", systemImage: "circle", summary: "Preview", platforms: [.iOS],
            allowsPreviewInteraction: true)
        let decoded = try JSONDecoder().decode(DemoDescriptor.self, from: JSONEncoder().encode(interactive))
        #expect(decoded.allowsPreviewInteraction && decoded == interactive)
    }
}
