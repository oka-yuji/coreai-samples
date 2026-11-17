import Testing
import VoxCPMCore

@Suite("VoxCPM2 noise")
struct VoxCPMNoiseTests {
    @Test("Seed 1708 noise matches the fixture noise bit for bit for patches 0 through 33")
    func seededNoise() throws {
        let noise = try VoxCPMReference.load().noise.map(VoxCPMReference.floats)
        try #require(noise.count == 34)
        var generator = VoxCPMNoise(seed: 1708)
        for (patch, expected) in noise.enumerated() {
            #expect(VoxCPMReference.bits(generator.nextPatch()) == VoxCPMReference.bits(expected), "patch \(patch)")
        }
    }
}
