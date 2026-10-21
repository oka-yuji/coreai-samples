import FeatureStore
import Foundation
import ImageDiagnosticFeature
import ImageGenerationCore
import ModelLibraryFeature
import Testing

struct ImageDiagnosticFeatureTests {
    @Test("Only the explicit diagnostic flag selects image execution and ordinary launches remain in Models")
    func launchSelection() {
        #expect(IOSDemoLaunch.resolve(arguments: ["app"]) == .models)
        #expect(
            IOSDemoLaunch.resolve(arguments: ["app", "--image-generation-check", "request.json"])
                == .imageGenerationDiagnostic("request.json"))
        #expect(
            IOSDemoLaunch.resolve(arguments: ["app", "--image-generation-check=request.json"])
                == .imageGenerationDiagnostic("request.json"))
    }

    @Test("Repeated stop requests keep the existing cleanup effect alive and old results cannot finish it")
    func repeatedStop() {
        var state = ImageDiagnosticState()
        let start = ImageDiagnosticFeature.reduce(&state, .start(URL(filePath: "/tmp/request.json")))
        #expect(start.count == 1)
        let original = state.generation
        let stop = ImageDiagnosticFeature.reduce(&state, .stop)
        #expect(stop == [.cancelAll, .run(.stop(original + 1), id: .stop)])
        #expect(ImageDiagnosticFeature.reduce(&state, .stop).isEmpty)
        let late = ImageDiagnosticOutcome(status: .completed, directory: "/tmp/result", message: "Done")
        #expect(ImageDiagnosticFeature.reduce(&state, .completed(original, late)).isEmpty)
        #expect(state.phase == .stopping)
        _ = ImageDiagnosticFeature.reduce(&state, .stopFailed(state.generation, "cleanup failed"))
        #expect(state.phase == .cleanupFailed)
        #expect(ImageDiagnosticFeature.reduce(&state, .start(URL(filePath: "/tmp/other.json"))).isEmpty)
        #expect(ImageDiagnosticFeature.reduce(&state, .stop).count == 2)
        _ = ImageDiagnosticFeature.reduce(&state, .stopped(state.generation, nil))
        #expect(state.phase == .finished)
    }

    @Test("A delayed cancellation is scheduled once only for the configured current operation")
    func delayedStop() throws {
        let plan = try JSONDecoder().decode(
            ImageDiagnosticStop.self,
            from: Data(#"{"component":"denoiser","operation":"inference","step":2,"delayMilliseconds":25}"#.utf8))
        var state = ImageDiagnosticState()
        _ = ImageDiagnosticFeature.reduce(&state, .start(URL(filePath: "/tmp/request.json")))
        let generation = state.generation
        _ = ImageDiagnosticFeature.reduce(&state, .configured(generation, plan))
        #expect(ImageDiagnosticFeature.reduce(&state, .progressed(generation, .running(.denoiser, step: 1))).isEmpty)
        #expect(
            ImageDiagnosticFeature.reduce(&state, .progressed(generation, .running(.denoiser, step: 2)))
                == [.run(.delayStop(generation, 25), id: .delay)])
        #expect(ImageDiagnosticFeature.reduce(&state, .progressed(generation, .running(.denoiser, step: 2))).isEmpty)
        #expect(ImageDiagnosticFeature.reduce(&state, .stopElapsed(generation - 1)).isEmpty)
        #expect(state.phase == .running)
        #expect(ImageDiagnosticFeature.reduce(&state, .stopElapsed(generation)).count == 2)
        #expect(state.phase == .stopping)
    }

    @Test("Completion cancels a pending timer and diagnostic failures cannot restart automatically")
    func finished() {
        var state = ImageDiagnosticState()
        _ = ImageDiagnosticFeature.reduce(&state, .start(URL(filePath: "/tmp/request.json")))
        let outcome = ImageDiagnosticOutcome(status: .completed, directory: "/tmp/result", message: "Done")
        #expect(ImageDiagnosticFeature.reduce(&state, .completed(state.generation, outcome)) == [.cancel(.delay)])
        #expect(state.phase == .finished)
        #expect(ImageDiagnosticFeature.reduce(&state, .stopElapsed(state.generation)).isEmpty)
        #expect(ImageDiagnosticFeature.reduce(&state, .start(URL(filePath: "/tmp/request.json"))).isEmpty)
    }
}
