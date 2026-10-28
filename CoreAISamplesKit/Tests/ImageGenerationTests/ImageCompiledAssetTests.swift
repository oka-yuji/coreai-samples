import CryptoKit
import Foundation
import ImageGenerationCore
import Testing

@testable import ImageGenerationRuntime

struct ImageCompiledAssetTests {
    @Test("A compiled component binds to its raw source and is passed through the existing pipeline")
    func boundComponent() async throws {
        let fixture = try compiledFixture()
        defer { fixture.remove() }
        let bundle = try ImageRuntimeBundle.load(fixture.resource, budget: fixture.budget)
        #expect(bundle.manifest.compiledAssets?["denoiser"]?.architecture == "h18p")
        let backend = ImageFakeBackend()
        let pipeline = fixture.pipeline(backend: backend, compiledArchitecture: "h18p")
        let result = try await pipeline.generate(fixture.request(), progress: { _ in })
        #expect(result.report.status == .completed)
        #expect(await backend.compiledAssets.count == 1)
        #expect(await backend.compiledAssets["denoiser/denoiser.aimodel"]?.asset == "compiled/denoiser.h18p.aimodelc")
    }

    @Test("AOT requests without an exact architecture match fail before any model loads", arguments: [false, true])
    func unavailableArchitecture(_ hasCompiled: Bool) async throws {
        let fixture = try hasCompiled ? compiledFixture() : ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        let pipeline = fixture.pipeline(backend: backend, compiledArchitecture: "h19p")
        await #expect(throws: ImageExecutionError.invalidResource) {
            try await pipeline.generate(fixture.request(), progress: { _ in })
        }
        #expect(await backend.history.isEmpty)
    }

    @Test("An inventory cannot authorize a compiled model from a different raw source", arguments: [false, true])
    func sourceMismatch(_ metadata: Bool) throws {
        let fixture = try compiledFixture { manifest, files in
            if metadata {
                files["compiled/denoiser.h18p.aimodelc/metadata.json"] =
                    try JSONSerialization.data(withJSONObject: ["sourceHash": String(repeating: "b", count: 64)])
            } else {
                var compiled = try #require(manifest["compiled_assets"] as? [String: [String: String]])
                compiled["denoiser"]?["source_sha256"] = String(repeating: "b", count: 64)
                manifest["compiled_assets"] = compiled
            }
        }
        defer { fixture.remove() }
        #expect(throws: ImageGenerationError.self) {
            try ImageRuntimeBundle.load(fixture.resource, budget: fixture.budget)
        }
    }

    @Test("Models import checks the compiled metadata source before marking a bundle ready")
    func importMetadataMismatch() async throws {
        let fixture = try compiledFixture { _, files in
            files["compiled/denoiser.h18p.aimodelc/metadata.json"] =
                try JSONSerialization.data(withJSONObject: ["sourceHash": String(repeating: "b", count: 64)])
        }
        defer { fixture.remove() }
        await #expect(throws: (any Error).self) { try await fixture.bundle.install(into: fixture.bundle.store()) }
    }

    @Test(
        "Unsafe paths unknown components and unlisted compiled metadata are refused",
        arguments: ["path", "kind", "architecture", "missing"])
    func invalidManifest(_ mutation: String) throws {
        let fixture = try compiledFixture { manifest, files in
            var compiled = try #require(manifest["compiled_assets"] as? [String: [String: String]])
            switch mutation {
            case "path": compiled["denoiser"]?["asset"] = "../outside.aimodelc"
            case "kind": compiled["unknown"] = compiled.removeValue(forKey: "denoiser")
            case "architecture": compiled["denoiser"]?["architecture"] = "../h18p"
            default: files.removeValue(forKey: "compiled/denoiser.h18p.aimodelc/metadata.json")
            }
            manifest["compiled_assets"] = compiled
        }
        defer { fixture.remove() }
        #expect(throws: ImageGenerationError.self) {
            try ImageRuntimeBundle.load(fixture.resource, budget: fixture.budget)
        }
    }

    @Test(
        "Modified missing and linked compiled files fail inventory validation",
        arguments: ["change", "missing", "link"])
    func corruptFiles(_ mutation: String) throws {
        let fixture = try compiledFixture()
        defer { fixture.remove() }
        let file = fixture.resource.directory.appending(path: "compiled/denoiser.h18p.aimodelc/resources.bin")
        if mutation == "change" {
            try Data("changed".utf8).write(to: file)
        } else {
            try FileManager.default.removeItem(at: file)
            if mutation == "link" {
                try FileManager.default.createSymbolicLink(
                    at: file, withDestinationURL: fixture.resource.directory.appending(path: "LICENSE"))
            }
        }
        #expect(throws: ImageExecutionError.invalidResource) {
            try ImageRuntimeBundle.load(fixture.resource, budget: fixture.budget)
        }
    }

    private func compiledFixture(
        mutate: (inout [String: Any], inout [String: Data]) throws -> Void = { _, _ in }
    ) throws -> ImageRuntimeFixture {
        try ImageRuntimeFixture { manifest, files in
            let raw = Data("raw model fixture".utf8)
            let hash = SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined()
            files["denoiser/denoiser.aimodel/main.mlirb"] = raw
            files["compiled/denoiser.h18p.aimodelc/resources.bin"] = Data("compiled model fixture".utf8)
            files["compiled/denoiser.h18p.aimodelc/metadata.json"] =
                try JSONSerialization.data(withJSONObject: ["sourceHash": hash.uppercased()])
            manifest["compiled_assets"] = [
                "denoiser": [
                    "asset": "compiled/denoiser.h18p.aimodelc", "architecture": "h18p", "source_sha256": hash,
                ]
            ]
            try mutate(&manifest, &files)
        }
    }
}
