import CoreGraphics
import Foundation
import ImageIO
import Testing
import VideoSearchFeature
import VideoSearchRuntime

private let embeddingBundle = (ProcessInfo.processInfo.environment["COREAI_TEST_BUNDLES"] ?? "")
    .split(separator: ":").map { URL(filePath: String($0)) }
    .first { FileManager.default.fileExists(atPath: $0.appending(path: "embedding-manifest.json").path()) }

@MainActor
@Suite("Video search with EmbeddingGemma 2", .serialized, .enabled(if: embeddingBundle != nil))
struct VideoSearchIntegrationTests {
    private struct Query: Decodable {
        let id: String
        let text: String
    }

    @Test(
        "VID-06 A fifteen-second video of the five bundled images ranks the tomato and white cheese scene between 12 and 15 seconds and folds its neighbors"
    )
    func findsTomatoAndCheese() async throws {
        let bundle = try #require(embeddingBundle)
        let resources = URL(filePath: #filePath).deletingLastPathComponent()
            .appending(path: "../../../CoreAISamples/Resources").standardized
        let names = ["01-oyakodon", "02-ginger-pork", "03-omelette", "04-gratin", "05-caprese"]
        let images = try names.map { try Self.image(resources.appending(path: "Recipes/\($0).png")) }
        let video = try await SyntheticVideo.write(width: 480, height: 640, frames: 150) { images[$0 / 30] }
        defer { SyntheticVideo.remove(video) }
        let queries = try JSONDecoder().decode(
            [Query].self, from: Data(contentsOf: resources.appending(path: "image-search-queries.json")))

        let service = LiveVideoSearchService()
        try await service.load(bundle)
        var progress: [[Int]] = []
        let indexing = ContinuousClock.now
        let index = try await service.index(video) { progress.append([$0, $1]) }
        let indexSeconds = Self.seconds(since: indexing)
        print("VID-06 indexed \(index.moments.count) frames of \(index.duration) s in \(indexSeconds) s")
        #expect(progress.last == [15, 15] && progress.count == 15)
        #expect(index.moments.count == 15 && abs(index.duration - 15) <= 0.2)

        for id in ["ja_food", "en_food"] {
            let query = try #require(queries.first { $0.id == id })
            let searching = ContinuousClock.now
            let matches = try await service.search(query.text)
            let searchSeconds = Self.seconds(since: searching)
            let folded = VideoSearchState.suppressNeighbors(matches, gap: VideoSearchState.minimumGap)
            let ranking = matches.prefix(5).map { "\($0.moment.time)@\($0.similarity)" }.joined(separator: " ")
            print("VID-06 \(id) searched in \(searchSeconds) s; top \(ranking); folded \(folded.count)")
            let best = try #require(matches.first)
            #expect((12.0..<15.0).contains(best.moment.time), "\(id) \(best.moment.time)")
            try #require(folded.count >= 2)
            #expect(folded.count < matches.count)
            #expect(abs(folded[0].moment.time - folded[1].moment.time) > VideoSearchState.minimumGap)
        }
        try await service.close()
    }

    private static func image(_ url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw CocoaError(.fileReadCorruptFile, userInfo: [NSURLErrorKey: url]) }
        return image
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = (ContinuousClock.now - start).components
        return Double(duration.seconds) + Double(duration.attoseconds) / 1e18
    }
}
