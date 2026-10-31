import Foundation
import ImageGenerationCore
import Testing

@MainActor
struct ImageDeviceMemoryTests {
    @Test("Calibration checkpoints require an explicit configuration opt-in")
    func checkpointConfiguration() throws {
        let original = Data(#"{"runID":"run","requests":["input"],"repetitions":5}"#.utf8)
        let enabled = Data(#"{"runID":"run","requests":["input"],"repetitions":5,"checkpoints":true}"#.utf8)
        let disabled = Data(#"{"runID":"run","requests":["input"],"repetitions":5,"checkpoints":false}"#.utf8)
        let decoder = JSONDecoder()
        #expect(try decoder.decode(ImageDeviceCalibrationTests.Configuration.self, from: original).checkpoints == nil)
        #expect(try decoder.decode(ImageDeviceCalibrationTests.Configuration.self, from: enabled).checkpoints == true)
        #expect(try decoder.decode(ImageDeviceCalibrationTests.Configuration.self, from: disabled).checkpoints == false)
    }

    @Test("A measurement write failure retains the generation and cleanup failures")
    func recordingFailure() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: folder) }
        let memory = point(footprint: 100, available: 900, peak: 800)
        let attempt = ImageDeviceCalibrationTests.Attempt(
            requestSHA256: "test", requestID: UUID(), started: Date(), startToImageSeconds: nil, execution: nil,
            error: "primary-probe", cleanupError: "cleanup-probe", cancelledAfterRelease: false, outputSHA256: nil,
            released: memory, releasedAfter500Milliseconds: memory, releasedAfter5Seconds: memory, windows: [])
        do {
            try ImageDeviceCalibrationTests.save(attempt, to: folder)
            Issue.record("Writing a measurement to a directory must fail")
        } catch let failure as ImageDeviceCalibrationTests.RecordingFailure {
            #expect(failure.primary == "primary-probe")
            #expect(failure.cleanup == "cleanup-probe")
            #expect(!failure.recording.isEmpty && !failure.cancelled)
        }
    }

    @Test("Calibration run names reject path traversal and invalid repetition counts")
    func configurationBoundary() throws {
        try ImageDeviceCalibrationTests.Configuration(
            runID: "native-complex-v1", requests: ["input"], repetitions: 5, checkpoints: nil
        )
        .validate()
        for name in ["", "old/../../tmp/escape", "..", "../run", "name.ext"] {
            #expect(throws: ImageExecutionError.invalidResource) {
                try ImageDeviceCalibrationTests.Configuration(
                    runID: name, requests: ["input"], repetitions: 1, checkpoints: nil
                )
                .validate()
            }
        }
        for count in [0, 21] {
            #expect(throws: ImageExecutionError.invalidResource) {
                try ImageDeviceCalibrationTests.Configuration(
                    runID: "run", requests: ["input"], repetitions: count, checkpoints: nil
                )
                .validate()
            }
        }
    }

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
