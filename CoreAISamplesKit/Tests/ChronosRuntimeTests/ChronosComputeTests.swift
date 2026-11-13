import CoreAI
import Testing

@testable import ChronosRuntime

struct ChronosComputeTests {
    @Test("B4 CPU-only specialization excludes other compute units and preserves reshape policy")
    func cpuOnly() {
        let options = ChronosNativePredictor.specializationOptions(for: .cpuOnly)
        #expect(options.allowedComputeUnitKinds == [.cpu])
        #expect(!options.expectFrequentReshapes)
    }

    @Test("B6 GPU preference preserves the existing Chronos specialization policy")
    func gpuPreferred() {
        let options = ChronosNativePredictor.specializationOptions(for: .gpuPreferred)
        #expect(options.preferredComputeUnitKind == .gpu)
        #expect(options.allowedComputeUnitKinds.contains(.gpu))
        #expect(!options.expectFrequentReshapes)
    }
}
