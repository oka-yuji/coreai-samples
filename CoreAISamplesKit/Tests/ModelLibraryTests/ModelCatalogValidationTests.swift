import Foundation
import ModelLibraryFeature
import Testing

struct ModelCatalogValidationTests {
    private let fixture = LibraryFixture()

    @Test(
        "Catalogs reject file entries with unsafe paths, negative sizes or malformed checksums",
        arguments: [
            ModelArtifactFile(path: "", bytes: 1, sha256: LibraryFixture.checksum),
            ModelArtifactFile(path: "/model.bin", bytes: 1, sha256: LibraryFixture.checksum),
            ModelArtifactFile(path: "../model.bin", bytes: 1, sha256: LibraryFixture.checksum),
            ModelArtifactFile(path: "bundle/../model.bin", bytes: 1, sha256: LibraryFixture.checksum),
            ModelArtifactFile(path: "bundle//model.bin", bytes: 1, sha256: LibraryFixture.checksum),
            ModelArtifactFile(path: "model.bin", bytes: -1, sha256: LibraryFixture.checksum),
            ModelArtifactFile(path: "model.bin", bytes: 1, sha256: LibraryFixture.checksum.uppercased()),
            ModelArtifactFile(path: "model.bin", bytes: 1, sha256: String(LibraryFixture.checksum.dropLast())),
        ])
    func malformedFiles(file: ModelArtifactFile) {
        #expect(throws: ModelCatalogError.invalidArtifact("bad")) {
            try ModelCatalog(
                models: [
                    fixture.model(
                        "model", artifacts: [fixture.artifact("bad", files: [file])], defaultArtifactID: "bad")
                ],
                demos: [fixture.demo("text")])
        }
    }

    @Test("Catalogs reject missing defaults, duplicate variants, unsafe roots and pinned distributions without files")
    func malformedVariants() throws {
        let demos = [fixture.demo("text")]
        let file = ModelArtifactFile(path: "model.bin", bytes: 1, sha256: LibraryFixture.checksum)
        #expect(throws: ModelCatalogError.invalidModel("model")) {
            try ModelCatalog(models: [fixture.model("model", artifacts: [], defaultArtifactID: "fp32")], demos: demos)
        }
        #expect(throws: ModelCatalogError.invalidModel("model")) {
            try ModelCatalog(
                models: [fixture.model("model", artifacts: [fixture.artifact("fp32")], defaultArtifactID: "int4")],
                demos: demos)
        }
        #expect(throws: ModelCatalogError.duplicateArtifact("fp32")) {
            try ModelCatalog(
                models: [
                    fixture.model("first", artifacts: [fixture.artifact("fp32")], defaultArtifactID: "fp32"),
                    fixture.model("second", artifacts: [fixture.artifact("fp32")], defaultArtifactID: "fp32"),
                ], demos: demos)
        }
        for invalid in [
            fixture.artifact("bad", root: "../escape"), fixture.artifact("bad", root: "/absolute"),
            fixture.artifact("bad", files: []),
            fixture.artifact("bad", files: [file, file]),
            ModelArtifact(id: "bad", title: "", contract: "fixture", platforms: [.iOS]),
            ModelArtifact(id: "bad", title: "Bad", contract: "fixture", platforms: []),
        ] {
            #expect(throws: ModelCatalogError.invalidArtifact("bad")) {
                try ModelCatalog(
                    models: [fixture.model("model", artifacts: [invalid], defaultArtifactID: "bad")], demos: demos)
            }
        }
        let accepted = [
            fixture.artifact("rooted", root: "bundle/v1"),
            ModelArtifact(id: "unconfigured", title: "Unconfigured", contract: "unconfigured", platforms: [.iOS]),
        ]
        let catalog = try ModelCatalog(
            models: [fixture.model("model", artifacts: accepted, defaultArtifactID: "unconfigured")], demos: demos)
        #expect(catalog.model("model")?.artifacts == accepted)
    }
}
