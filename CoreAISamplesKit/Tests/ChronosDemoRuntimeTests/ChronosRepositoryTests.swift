import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

struct ChronosRepositoryTests {
    @Test("D1 All thirty-nine fixed frames preserve the exact scene, area and origin request")
    func bundled() async throws {
        let repository = try ChronosRepository.bundled()
        let catalog = try await repository.load()
        #expect(catalog.scenarios.map(\.id) == ["friday-evening", "weekday-morning", "saturday-night"])
        #expect(catalog.areas.count == 20 && catalog.timeZone == "America/New_York")
        #expect(catalog.historySteps == 1008 && catalog.truthSteps == 16 && catalog.intervalSeconds == 1800)
        for scene in catalog.scenarios {
            for step in 0...12 {
                let request = try #require(catalog.request(scenarioID: scene.id, step: step, mode: .precomputed))
                let frame = try await repository.cached(request)
                #expect(frame.scenarioID == scene.id && frame.originStep == step && frame.mode == .precomputed)
                #expect(frame.areas.count == 20 && frame.processingSeconds == nil && frame.identity == nil)
                for series in request.series {
                    let history = try #require(scene.history[series.areaID])
                    let truth = try #require(scene.truth[series.areaID])
                    #expect(series.values == Array((history + truth.prefix(step)).suffix(1008)))
                }
            }
        }
        let first = try #require(catalog.request(scenarioID: "friday-evening", step: 0, mode: .precomputed))
        let result = try await repository.cached(first)
        #expect(result.area("48")?.median[0] == Float(72.50890350341797))
    }

    @Test("D4 Unknown or modified cache requests fail without substituting another frame")
    func invalidRequests() async throws {
        let repository = try ChronosRepository.bundled()
        let catalog = try await repository.load()
        let valid = try #require(catalog.request(scenarioID: "friday-evening", step: 0, mode: .precomputed))
        for request in [
            ChronosForecastRequest(scenarioID: "unknown", originStep: 0, mode: .precomputed, series: valid.series),
            ChronosForecastRequest(
                scenarioID: valid.scenarioID, originStep: 13, mode: .precomputed, series: valid.series),
            ChronosForecastRequest(scenarioID: valid.scenarioID, originStep: 0, mode: .live, series: valid.series),
            ChronosForecastRequest(scenarioID: valid.scenarioID, originStep: 0, mode: .precomputed, series: []),
        ] {
            await #expect(throws: (any Error).self) { try await repository.cached(request) }
        }
    }

    @Test(
        "D6 Missing, extra, symlinked and corrupted resources fail validation",
        arguments: ["missing", "extra", "link", "cache", "source", "manifest", "geometry"])
    func corruption(_ kind: String) async throws {
        let source = try ChronosRepository.bundledDirectory()
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.copyItem(at: source, to: root)
        defer { try? FileManager.default.removeItem(at: root) }
        switch kind {
        case "missing": try FileManager.default.removeItem(at: root.appending(path: "forecasts.json"))
        case "extra": try Data().write(to: root.appending(path: "extra.json"))
        case "link":
            try FileManager.default.removeItem(at: root.appending(path: "forecasts.json"))
            try FileManager.default.createSymbolicLink(
                at: root.appending(path: "forecasts.json"), withDestinationURL: source.appending(path: "forecasts.json")
            )
        default:
            let name =
                [
                    "cache": "forecasts.json", "source": "friday-evening.input.json", "manifest": "manifest.json",
                    "geometry": "city.json",
                ][kind] ?? "forecasts.json"
            var data = try Data(contentsOf: root.appending(path: name))
            data.append(32)
            try data.write(to: root.appending(path: name))
        }
        await #expect(throws: (any Error).self) { try await ChronosRepository(directory: root).load() }
    }
}
