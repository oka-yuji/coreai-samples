import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

struct VoxCPMReferenceRuntimeTests {
    @Test("VRW-01 The opened reference model accepts a voice through the SpeechGenerating requirement")
    func protocolWitness() async throws {
        let driver = FakeVoxCPMDriver()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies(profile: .referenceFP32))
        let service: any SpeechGenerating = runtime
        let model = try await service.open(URL(filePath: "/unused-model"))
        #expect(model.supportsReferenceAudio)
        let reference = try SpeechReferenceAudio(
            samples: [Float](repeating: 0.25, count: 80_000), source: .recording, format: .voxcpm2)
        let result = try await service.generate(text: "ab", seed: 1708, reference: reference)
        #expect(!result.wav.isEmpty)
        #expect(await driver.factoryCount == 1)
        await service.release()
    }
}
