import Foundation
import ImageIO
import ObjectFinderCore
import ObjectFinderRuntime

@main
enum ObjectFinderCheck {
    struct Plan: Decodable {
        struct Sample: Decodable {
            let id: String
            let image: String
            let query: String
        }
        let minimumSimilarity: Double
        let mode: ObjectSearchMode?
        let samples: [Sample]
    }

    struct Record: Encodable {
        let id: String
        let width: Int
        let height: Int
        let seconds: Double
        let candidates: [ObjectCandidate]
        let selected: [ObjectCandidate]
    }

    static func main() async {
        do { try await run() } catch {
            FileHandle.standardError.write(Data("Object finder check failed: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 4 else {
            throw CocoaError(.fileReadInvalidFileName)
        }
        let output = URL(filePath: arguments[3])
        guard !FileManager.default.fileExists(atPath: output.path) else { throw CocoaError(.fileWriteFileExists) }
        let plan = try JSONDecoder().decode(Plan.self, from: Data(contentsOf: URL(filePath: arguments[2])))
        let finder = ObjectRegionFinder()
        var records: [Record] = []
        for sample in plan.samples {
            try await finder.prepare(folder: URL(filePath: arguments[1]), query: sample.query)
            guard let source = CGImageSourceCreateWithURL(URL(filePath: sample.image) as CFURL, nil),
                let image = CGImageSourceCreateThumbnailAtIndex(
                    source, 0,
                    [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceCreateThumbnailWithTransform: true,
                        kCGImageSourceThumbnailMaxPixelSize: 1280,
                    ] as CFDictionary)
            else { throw ObjectFinderError.invalidImage }
            let start = ContinuousClock.now
            let candidates = try await finder.candidates(in: image, mode: plan.mode ?? .standard)
            let duration = (ContinuousClock.now - start).components
            records.append(
                Record(
                    id: sample.id, width: image.width, height: image.height,
                    seconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18,
                    candidates: candidates,
                    selected: ObjectCandidateSelection.select(candidates, minimumSimilarity: plan.minimumSimilarity)))
        }
        try await finder.close()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(records).write(to: output, options: .atomic)
    }
}
