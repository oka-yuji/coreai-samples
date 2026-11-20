import SpeechCore
import Testing

@testable import SpeechRuntime

struct VoxCPMReferencePipelineTests {
    @Test("VRP-01 A target of 111 tokens with a five-second reference is rejected before a state or model call")
    func targetAboveReferenceLimit() async throws {
        let contract = try VoxCPMPipelineTests.contract(.referenceFP32)
        let log = FakeVoxCPMLog()
        let pipeline = VoxCPMPipeline(
            components: FakeVoxCPMComponents(log: log, contract: contract), contract: contract)
        let voice = try SpeechReferenceAudio(
            samples: [Float](repeating: 0.25, count: 80_000), source: .recording, format: .voxcpm2)
        let ids = [Int32](repeating: 11, count: 111) + [101]
        await #expect(throws: VoxCPMGenerationError.textTooLong(111)) {
            try await pipeline.generate(ids: ids, seed: 1708, reference: voice)
        }
        #expect(log.events.isEmpty && log.calls.isEmpty)
    }
}
