import Foundation
import ImageGenerationCore
import ImageGenerationRuntime
import ModelLibraryFeature
import Testing

#if os(iOS)
    import UIKit
#endif

@MainActor
struct ImageDeviceCalibrationTests {
    struct Configuration: Decodable {
        let runID: String
        let requests: [String]
        let repetitions: Int
        let checkpoints: Bool?

        func validate() throws {
            guard (1...20).contains(repetitions), (1...2).contains(requests.count), !runID.isEmpty,
                runID.utf8.allSatisfy({ $0 == 45 || (48...57).contains($0) || (97...122).contains($0) })
            else { throw ImageExecutionError.invalidResource }
        }
    }

    struct Attempt: Encodable {
        let requestSHA256: String
        let requestID: UUID
        let started: Date
        let startToImageSeconds: Double?
        let execution: ImageExecutionReport?
        let error: String?
        let cleanupError: String?
        let cancelledAfterRelease: Bool
        let outputSHA256: String?
        let released: ImageMemoryObservation
        let releasedAfter500Milliseconds: ImageMemoryObservation
        let releasedAfter5Seconds: ImageMemoryObservation
        let windows: [ImageDeviceMemoryObserver.Window]
        let scope = "Hosted main-app service, sampled progress intervals; lifetime peak is cumulative, not phase peak"
    }

    struct RecordingFailure: Error, CustomStringConvertible {
        let primary: String?
        let cleanup: String?
        let recording: String
        let cancelled: Bool

        var description: String {
            "generation=\(primary ?? "none"); cleanup=\(cleanup ?? "none"); recording=\(recording); cancelled=\(cancelled)"
        }
    }

    static func save(_ attempt: Attempt, to url: URL) throws {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(attempt).write(to: url, options: .withoutOverwriting)
        } catch {
            throw RecordingFailure(
                primary: attempt.error, cleanup: attempt.cleanupError, recording: String(describing: error),
                cancelled: attempt.cancelledAfterRelease || Task.isCancelled)
        }
    }

    #if os(iOS)
        @Test(
            "Explicit physical-device calibration records sequential real-model generation and editing",
            .enabled(if: ProcessInfo.processInfo.environment["COREAI_IMAGE_CALIBRATION_CONFIG"] != nil),
            .timeLimit(.minutes(120)))
        func consecutive() async throws {
            try #require(UIApplication.shared.applicationState == .active)
            let idleTimerWasDisabled = UIApplication.shared.isIdleTimerDisabled
            UIApplication.shared.isIdleTimerDisabled = true
            defer { UIApplication.shared.isIdleTimerDisabled = idleTimerWasDisabled }
            let path = try #require(ProcessInfo.processInfo.environment["COREAI_IMAGE_CALIBRATION_CONFIG"])
            let url = URL(filePath: path, relativeTo: .documentsDirectory)
            let raw = try ImageDeviceCalibrationInput.read(url, maximum: 1_048_576)
            let config = try JSONDecoder().decode(Configuration.self, from: raw)
            try config.validate()
            let display = try ImageCalibrationProgress()
            defer { display.close() }
            let output = URL.documentsDirectory.appending(path: "image-calibration-" + config.runID)
            try #require(!FileManager.default.fileExists(atPath: output.path))
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
            try raw.write(to: output.appending(path: "configuration.json"), options: .withoutOverwriting)
            for iteration in 0..<config.repetitions {
                for (index, request) in config.requests.enumerated() {
                    display.begin(
                        iteration * config.requests.count + index + 1, total: config.repetitions * config.requests.count
                    )
                    let requestURL = URL(filePath: request, relativeTo: .documentsDirectory)
                    try await run(
                        requestURL, output: output.appending(path: "\(iteration)-\(index)"),
                        checkpoints: config.checkpoints == true, display: display)
                }
            }
            try Task.checkCancellation()
            try Data("completed\n".utf8).write(
                to: output.appending(path: "completed.txt"), options: .withoutOverwriting)
        }

        private func run(
            _ url: URL, output: URL, checkpoints: Bool, display: ImageCalibrationProgress
        ) async throws {
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
            let raw = try ImageDeviceCalibrationInput.read(url, maximum: 1_048_576)
            let input = try JSONDecoder().decode(ImageDeviceCalibrationInput.self, from: raw)
            let (resource, request) = try input.prepare(at: url)
            try raw.write(to: output.appending(path: "request.json"), options: .withoutOverwriting)
            let service = LiveImageGenerationService(
                session: ModelDemoSession(
                    id: LibraryTicket(owner: UUID(), number: 1),
                    selection: DemoSelection(modelID: resource.modelID, demoID: "image-generation"), resource: resource),
                budget: input.budget, diagnosticDirectory: checkpoints ? output : nil,
                isPermitted: { UIApplication.shared.applicationState == .active })
            let observer = ImageDeviceMemoryObserver()
            observer.transition("start")
            async let monitoring: Void = observer.watch()
            let started = Date()
            let clock = ContinuousClock.now
            var seconds: Double?
            var failure: String?
            var cleanup: String?
            var outputHash: String?
            do {
                let result = try await service.generate(request, trace: nil) { value in
                    observer.transition(String(describing: value.phase))
                    display.update(value.phase)
                }
                let asset = try await ImageGenerationMedia().asset(result, request: request)
                seconds = (ContinuousClock.now - clock) / .seconds(1)
                try asset.png.write(to: output.appending(path: "output.png"), options: .withoutOverwriting)
                let pixels = Data(result.pixels.rgba)
                try pixels.write(to: output.appending(path: "output.rgba"), options: .withoutOverwriting)
                outputHash = ImageDeviceCalibrationInput.digest(pixels)
            } catch { failure = String(describing: error) }
            do { try await service.release() } catch { cleanup = String(describing: error) }
            observer.transition(cleanup == nil ? "released" : "release-unconfirmed")
            let released = ImageDeviceMemory.capture()
            let after500: ImageMemoryObservation
            let after5: ImageMemoryObservation
            await withTaskCancellationShield {
                do { try await Task.sleep(for: .milliseconds(500)) } catch {}
            }
            after500 = ImageDeviceMemory.capture()
            await withTaskCancellationShield {
                do { try await Task.sleep(for: .milliseconds(4_500)) } catch {}
            }
            after5 = ImageDeviceMemory.capture()
            let windows = observer.finish()
            await monitoring
            let attempt = Attempt(
                requestSHA256: ImageDeviceCalibrationInput.digest(raw), requestID: request.id, started: started,
                startToImageSeconds: seconds, execution: await service.latestReport(), error: failure,
                cleanupError: cleanup, cancelledAfterRelease: Task.isCancelled, outputSHA256: outputHash,
                released: released,
                releasedAfter500Milliseconds: after500, releasedAfter5Seconds: after5, windows: windows)
            try Self.save(attempt, to: output.appending(path: "measurement.json"))
            try Task.checkCancellation()
            try #require(
                failure == nil && cleanup == nil, "Calibration attempt failed: \(failure ?? cleanup ?? "unknown")")
        }
    #endif
}
