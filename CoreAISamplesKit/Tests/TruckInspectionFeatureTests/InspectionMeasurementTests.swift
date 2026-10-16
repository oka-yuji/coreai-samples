import Foundation
import Testing
import TruckInspectionCore
import TruckInspectionFeature
import TruckInspectionRuntime

struct InspectionMeasurementTests {
    @Test("Measurement uses twenty warm samples with an even-sample median and nearest-rank p95")
    func statistics() throws {
        let result = try #require(InspectionMeasurement(warmupMilliseconds: 99, samples: (1...20).map(Double.init)))
        #expect(result.medianMilliseconds == 10.5)
        #expect(result.p95Milliseconds == 19)
        #expect(result.samples.count == 20)
        #expect(InspectionMeasurement(warmupMilliseconds: 0, samples: [1]) == nil)
        #expect(InspectionMeasurement(warmupMilliseconds: .nan, samples: Array(repeating: 1, count: 20)) == nil)
    }

    @Test("Measurement leaves inspection records unchanged and ignores cancelled results")
    func isolation() throws {
        var state = InspectionState(catalog: try InspectionCatalogLoader.load())
        state.active = true
        state.modelInfo = InspectionModelInfo(
            folder: URL(filePath: "/asset"), revision: "test", milliseconds: 1, computePreference: "gpu")
        _ = TruckInspectionFeature.reduce(&state, .start)
        _ = TruckInspectionFeature.reduce(&state, .choose(.attention))
        _ = TruckInspectionFeature.reduce(&state, .stop)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        let records = state.records
        let commands = TruckInspectionFeature.reduce(&state, .measure)
        #expect(commands.count == 1)
        #expect(state.phase == .measuring)
        #expect(TruckInspectionFeature.reduce(&state, .start).isEmpty)
        let id = state.generation
        _ = TruckInspectionFeature.reduce(&state, .stop)
        let result = try #require(InspectionMeasurement(warmupMilliseconds: 1, samples: Array(repeating: 1, count: 20)))
        #expect(TruckInspectionFeature.reduce(&state, .measured(id, result)).isEmpty)
        #expect(state.measurement == nil)
        #expect(state.records == records)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        _ = TruckInspectionFeature.reduce(&state, .measure)
        _ = TruckInspectionFeature.reduce(&state, .measured(state.generation, result))
        #expect(state.phase == .idle)
        #expect(state.measurement == result)
        #expect(state.records == records)
        _ = TruckInspectionFeature.reduce(&state, .modelChosen(URL(filePath: "/another")))
        #expect(state.measurement == nil)
    }

    @MainActor
    @Test("Measurement runs one warmup and twenty model calls without substituting exact matching")
    func modelCalls() async throws {
        let catalog = try InspectionCatalogLoader.load()
        let item = try #require(catalog.items.first)
        let model = InspectionModelFake()
        let runner = TruckInspectionRunner(model: model)
        var result: InspectionMeasurement?
        await runner.run(.measure(1, "fixed short answer", item, catalog)) {
            if case .measured(1, let measured) = $0 { result = measured }
        }
        #expect(model.classifiedTexts == Array(repeating: "fixed short answer", count: 21))
        #expect(result?.samples.count == 20)
    }
}
