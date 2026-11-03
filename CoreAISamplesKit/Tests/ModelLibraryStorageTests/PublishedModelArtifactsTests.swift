import Foundation
import ModelLibraryStorage
import Testing

@testable import ModelLibraryFeature

struct PublishedModelArtifactsTests {
    @Test("LIB-02 Shared artifacts built for iOS and the Mac match in everything except their platforms")
    func sharedArtifacts() throws {
        let catalog = try IOSDemoCatalog.make(referenceSpeechArtifact: IrodoriModelArtifact.makeReferenceDownload())
        let shared = [
            (
                PublishedModelArtifacts.glinerFP32(platforms: [.iOS]),
                PublishedModelArtifacts.glinerFP32(platforms: [.macOS])
            ),
            (
                PublishedModelArtifacts.embeddingFP16(platforms: [.iOS]),
                PublishedModelArtifacts.embeddingFP16(platforms: [.macOS])
            ),
            (
                PublishedModelArtifacts.embeddingPTQ(platforms: [.iOS]),
                PublishedModelArtifacts.embeddingPTQ(platforms: [.macOS])
            ),
            (
                try IrodoriModelArtifact.makeReferenceDownload(),
                try IrodoriModelArtifact.makeReferenceDownload(platforms: [.macOS])
            ),
        ]
        for (phone, mac) in shared {
            #expect(catalog.model(containing: phone.id)?.artifact(phone.id) == phone)
            #expect(phone.platforms == [.iOS] && mac.platforms == [.macOS])
            #expect(mac.files == phone.files && !mac.files.isEmpty)
            #expect(mac.download == phone.download && mac.download?.isValid == true)
            #expect(mac.upstream == phone.upstream && mac.contract == phone.contract)
            #expect(
                ModelArtifact(
                    id: mac.id, title: mac.title, contract: mac.contract, platforms: phone.platforms,
                    sizeBytes: mac.sizeBytes, download: mac.download, upstream: mac.upstream, files: mac.files,
                    visibility: mac.visibility, localBundle: mac.localBundle) == phone)
        }
    }
}
