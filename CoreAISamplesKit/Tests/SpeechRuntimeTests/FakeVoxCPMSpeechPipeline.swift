@testable import SpeechRuntime

struct FakeVoxCPMSpeechPipeline: VoxCPMSpeechPipeline {
    let lifetime: PipelineLifetime
    let driver: FakeVoxCPMDriver
    @concurrent
    func generate(ids: [Int32], seed: UInt64) async throws -> VoxCPMSpeech {
        try await driver.generate(ids: ids, seed: seed)
    }
}
