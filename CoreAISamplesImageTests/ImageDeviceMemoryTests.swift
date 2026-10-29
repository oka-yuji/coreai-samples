import ImageGenerationCore
import Testing

@MainActor
struct ImageDeviceMemoryTests {
    @Test("Sampled memory maxima stay distinct from cumulative kernel peak and missing samples")
    func sampledIntervals() {
        let first = point(footprint: 100, available: 900, peak: 800)
        var window = ImageDeviceMemoryObserver.Window(
            label: "loading", startedSeconds: 1, before: first, endedSeconds: 1, after: first)
        window.append(first, at: 1)
        window.append(point(footprint: 150, available: 850, peak: 800), at: 1.01)
        window.append(point(footprint: nil, available: 875, peak: nil), at: 1.05)
        #expect(window.sampleCount == 3)
        #expect(window.invalidSampleCount == 1)
        #expect(window.sampledFootprintMaximum == 150)
        #expect(window.kernelLifetimePeakMaximum == 800)
        #expect(window.minimumAvailableBytes == 850)
        #expect(abs(window.maximumGapSeconds - 0.04) < 0.000001)
        #expect(window.before == first)
        #expect(window.after.footprintBytes == nil)
    }

    private func point(footprint: UInt64?, available: UInt64, peak: UInt64?) -> ImageMemoryObservation {
        ImageMemoryObservation(
            scope: .processDirtyBudget, availableBytes: available, footprintBytes: footprint,
            residentBytes: 200, kernelLifetimePeakBytes: peak, visibleMetalBytes: 100, physicalMemoryBytes: 2_000)
    }
}
