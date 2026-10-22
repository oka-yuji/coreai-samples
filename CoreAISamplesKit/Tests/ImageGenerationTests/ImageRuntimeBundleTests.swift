import Foundation
import ImageGenerationCore
import ModelLibraryFeature
import Testing

@testable import ImageGenerationRuntime

struct ImageRuntimeBundleTests {
    @Test("The runtime accepts only the pinned complete five-component local bundle")
    func completeBundle() throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let bundle = try ImageRuntimeBundle.load(fixture.resource, budget: fixture.budget)
        #expect(bundle.manifestSHA256 == fixture.resource.revision)
        #expect(bundle.profile.id == fixture.budget.profileID)
        #expect(bundle.scaling.mean == [Float](repeating: 0, count: 64))
    }

    @Test(
        "Changed assets missing required files and unexpected paths fail before Core AI loads",
        arguments: ["changed", "missing", "extra-file", "extra-folder", "link"])
    func corruptBundle(_ mutation: String) throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let root = fixture.resource.directory
        switch mutation {
        case "changed": try Data("changed".utf8).write(to: root.appending(path: "LICENSE"))
        case "missing": try FileManager.default.removeItem(at: root.appending(path: "config/vae/config.json"))
        case "extra-file": try Data().write(to: root.appending(path: "extra"))
        case "extra-folder":
            try FileManager.default.createDirectory(
                at: root.appending(path: "extra"), withIntermediateDirectories: false)
        case "link":
            try FileManager.default.removeItem(at: root.appending(path: "LICENSE"))
            try FileManager.default.createSymbolicLink(
                at: root.appending(path: "LICENSE"), withDestinationURL: root.appending(path: "NOTICE.txt"))
        default: throw ImageTestError.missingEvent
        }
        #expect(throws: ImageExecutionError.invalidResource) {
            try ImageRuntimeBundle.load(fixture.resource, budget: fixture.budget)
        }
    }

    @Test(
        "Component headers are read beside each asset in the producer and nested layouts", arguments: ["", "models"])
    func headersBesideAssets(_ folder: String) throws {
        let fixture = try ImageRuntimeFixture(assetFolder: folder)
        defer { fixture.remove() }
        let bundle = try ImageRuntimeBundle.load(fixture.resource, budget: fixture.budget)
        #expect(bundle.profile.components["text"]?.asset == (folder.isEmpty ? "" : folder + "/") + "text/text.aimodel")
    }

    @Test("A listed component header that is not beside its asset is refused")
    func headerAwayFromAsset() throws {
        let fixture = try ImageRuntimeFixture(assetFolder: "models", headersBesideAssets: false)
        defer { fixture.remove() }
        let error = #expect(throws: CocoaError.self) {
            try ImageRuntimeBundle.load(fixture.resource, budget: fixture.budget)
        }
        #expect(error?.code == .fileReadNoSuchFile)
    }

    @Test(
        "AppleDouble and hidden files outside the inventory fail before Core AI loads",
        arguments: ["._unexpected", "config/._unexpected", ".DS_Store", "config/vae/.DS_Store"])
    func hiddenFiles(_ name: String) throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        try Data("hidden".utf8).write(to: fixture.resource.directory.appending(path: name))
        #expect(throws: ImageExecutionError.invalidResource) {
            try ImageRuntimeBundle.load(fixture.resource, budget: fixture.budget)
        }
    }

    @Test("A budget for another profile or manifest cannot select a model by path")
    func budgetIdentity() throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        for (profile, hash) in [
            ("other", fixture.resource.revision), (fixture.budget.profileID, String(repeating: "a", count: 64)),
        ] {
            let budget = try ImageMemoryBudget(
                profileID: profile, manifestSHA256: hash, components: fixture.budget.components)
            #expect(throws: ImageExecutionError.invalidResource) {
                try ImageRuntimeBundle.load(fixture.resource, budget: budget)
            }
        }
    }
}
