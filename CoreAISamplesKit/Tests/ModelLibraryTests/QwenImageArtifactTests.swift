import CryptoKit
import Foundation
import ModelLibraryFeature
import Testing

struct QwenImageArtifactTests {
    private struct Manifest: Decodable {
        let assets: [String: File]

        struct File: Decodable {
            let bytes: Int64
            let sha256: String
        }
    }

    @Test("The local Qwen artifact binds every file to the accepted raw manifest without a download URL")
    func inventory() throws {
        let url = try #require(
            Bundle.module.url(
                forResource: "qwen-native-complex-bundle", withExtension: "json", subdirectory: "Resources"))
        let data = try Data(contentsOf: url)
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(hash == "8b9e34748d760fa9ec7762f5064652281c80effb19c692bcdbe8ef1e521d59ad")
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        let artifact = QwenImage21Artifact.nativeComplex
        #expect(artifact.sourceKind == .localManifest && artifact.download == nil && artifact.isSourceValid)
        #expect(artifact.localBundle?.manifestSHA256 == hash)
        #expect(artifact.files.count == 51 && Set(artifact.files.map(\.path)).count == 51)
        #expect(artifact.sizeBytes == 13_783_495_099)
        #expect(artifact.files.reduce(0) { $0 + $1.bytes } == artifact.sizeBytes)
        let declared = Dictionary(uniqueKeysWithValues: artifact.files.map { ($0.path, $0) })
        #expect(
            declared["bundle.json"] == ModelArtifactFile(path: "bundle.json", bytes: Int64(data.count), sha256: hash))
        #expect(Set(declared.keys) == Set(manifest.assets.keys).union(["bundle.json"]))
        for (path, file) in manifest.assets {
            #expect(declared[path] == ModelArtifactFile(path: path, bytes: file.bytes, sha256: file.sha256))
        }
        let catalog = try IOSDemoCatalog.make(localImageArtifacts: [artifact])
        #expect(catalog.model("qwen-image21-turbo")?.artifacts == [artifact])
    }
}
