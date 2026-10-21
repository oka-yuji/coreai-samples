import Foundation
import ImageGenerationCore
import ImageGenerationFeature
import Testing

struct ImageGenerationFeatureTests {
    private func state(supported: Bool = true) -> ImageGenerationState {
        var state = ImageGenerationState(owner: UUID(), availability: supported ? .ready : .unsupported)
        _ = ImageGenerationFeature.reduce(&state, .appeared)
        _ = ImageGenerationFeature.reduce(&state, .setPrompt("A blue boat"))
        return state
    }

    private func asset(id: UUID, seed: UInt64) -> ImageGeneratedAsset {
        var report = ImageExecutionReport(
            requestID: id, profileID: "fixture", manifestSHA256: String(repeating: "a", count: 64),
            noiseSource: ImageGenerationNoise.version, diagnosticsEnabled: false)
        report.status = .completed
        return ImageGeneratedAsset(id: id, png: Data([1]), seed: seed, report: report)
    }

    @Test("Appearing unsupported resources and invalid inputs never schedule model generation")
    func availability() {
        var state = ImageGenerationState(owner: UUID(), availability: .ready)
        #expect(ImageGenerationFeature.reduce(&state, .appeared).isEmpty)
        #expect(ImageGenerationFeature.reduce(&state, .generate(id: UUID(), randomSeed: 3)).isEmpty)
        state = self.state(supported: false)
        #expect(ImageGenerationFeature.reduce(&state, .generate(id: UUID(), randomSeed: 3)).isEmpty)
        state = self.state()
        _ = ImageGenerationFeature.reduce(&state, .setSeed("-1"))
        #expect(!state.seedIsValid)
        #expect(ImageGenerationFeature.reduce(&state, .generate(id: UUID(), randomSeed: 3)).isEmpty)
        _ = ImageGenerationFeature.reduce(&state, .setSeed(String(UInt64.max)))
        #expect(!state.canGenerate)
    }

    @Test("The first chosen seed is retained after failure and reused by retry")
    func stableRetrySeed() throws {
        var state = state()
        let firstID = UUID()
        let ticket = ImageGenerationTicket(owner: state.owner, sequence: state.sequence + 1)
        let request = ImageGenerationRequest(id: firstID, prompt: state.prompt, seed: 123)
        #expect(
            ImageGenerationFeature.reduce(&state, .generate(id: firstID, randomSeed: 123)) == [
                .run(.generate(ticket, request), id: .operation)
            ])
        #expect(state.seed == "123")
        _ = ImageGenerationFeature.reduce(&state, .failed(ticket, .generationFailed))
        #expect(state.prompt == "A blue boat" && state.seed == "123")
        let secondID = UUID()
        let next = ImageGenerationTicket(owner: state.owner, sequence: state.sequence + 1)
        #expect(
            ImageGenerationFeature.reduce(&state, .generate(id: secondID, randomSeed: 456)) == [
                .run(
                    .generate(next, ImageGenerationRequest(id: secondID, prompt: state.prompt, seed: 123)),
                    id: .operation)
            ])
    }

    @Test("Stop cancels the operation but only a matching cleanup acknowledgement permits another run")
    func stopAndOldCallbacks() throws {
        var state = state()
        let id = UUID()
        _ = ImageGenerationFeature.reduce(&state, .generate(id: id, randomSeed: 12))
        let original = try #require(state.operation)
        let stopped = ImageGenerationTicket(owner: state.owner, sequence: state.sequence + 1)
        #expect(ImageGenerationFeature.reduce(&state, .stop) == [.cancel(.operation), .run(.stop(stopped), id: .stop)])
        #expect(!state.canGenerate)
        _ = ImageGenerationFeature.reduce(&state, .generated(original, asset(id: id, seed: 12)))
        _ = ImageGenerationFeature.reduce(&state, .failed(original, .generationFailed))
        _ = ImageGenerationFeature.reduce(&state, .stopped(original))
        #expect(state.phase == .stopping && state.result == nil && state.failure == nil)
        _ = ImageGenerationFeature.reduce(&state, .stopFailed(stopped))
        #expect(state.phase == .cleanupFailed && !state.canGenerate)
        _ = ImageGenerationFeature.reduce(&state, .stop)
        _ = ImageGenerationFeature.reduce(&state, .stopped(try #require(state.operation)))
        #expect(state.phase == .idle && state.canGenerate)
    }

    @Test("Input changes and returning to a previous binding reject old image results")
    func bindingAndInputChanges() throws {
        var first = state()
        let id = UUID()
        _ = ImageGenerationFeature.reduce(&first, .generate(id: id, randomSeed: 9))
        let ticket = try #require(first.operation)
        _ = ImageGenerationFeature.reduce(&first, .generated(ticket, asset(id: id, seed: 9)))
        #expect(first.result != nil)
        _ = ImageGenerationFeature.reduce(&first, .setPrompt("A red boat"))
        #expect(first.result == nil)
        _ = ImageGenerationFeature.reduce(&first, .generated(ticket, asset(id: id, seed: 9)))
        #expect(first.result == nil)
        var second = ImageGenerationState(owner: UUID(), availability: .ready, retaining: first)
        _ = ImageGenerationFeature.reduce(&second, .appeared)
        _ = ImageGenerationFeature.reduce(&second, .generate(id: id, randomSeed: 10))
        _ = ImageGenerationFeature.reduce(&second, .generated(ticket, asset(id: id, seed: 9)))
        #expect(second.result == nil && second.prompt == "A red boat" && second.seed == "9")
    }

    @Test("An imported image is not used until its exact crop preview is confirmed")
    func cropConfirmation() throws {
        var state = state()
        let path = URL(filePath: "/fixture/image.png")
        _ = ImageGenerationFeature.reduce(&state, .chooseReference(path))
        let ticket = try #require(state.operation)
        let pixels = try ImageGenerationPixels(
            width: 512, height: 512, rgba: [UInt8](repeating: 0, count: 512 * 512 * 4))
        let preview = ImageReferencePreview(id: ticket, thumbnail: Data([1]), cropped: Data([2]), pixels: pixels)
        _ = ImageGenerationFeature.reduce(&state, .referencePrepared(ticket, preview))
        #expect(state.phase == .confirmingReference && state.reference == nil && !state.canGenerate)
        _ = ImageGenerationFeature.reduce(
            &state, .confirmReference(ImageGenerationTicket(owner: UUID(), sequence: ticket.sequence)))
        #expect(state.reference == nil)
        _ = ImageGenerationFeature.reduce(&state, .confirmReference(ticket))
        #expect(state.reference == preview && state.proposedReference == nil && state.canGenerate)
    }

    @Test("Dismissing the save panel does not acknowledge a pending write or allow a duplicate export")
    func exportLifetime() throws {
        var state = state()
        let id = UUID()
        _ = ImageGenerationFeature.reduce(&state, .generate(id: id, randomSeed: 1))
        _ = ImageGenerationFeature.reduce(&state, .generated(try #require(state.operation), asset(id: id, seed: 1)))
        let exportID = UUID()
        _ = ImageGenerationFeature.reduce(&state, .prepareExport(exportID))
        let ticket = try #require(state.export)
        _ = ImageGenerationFeature.reduce(&state, .exportDismissed(ticket))
        #expect(!state.editable && !state.canExport && !state.canGenerate)
        _ = ImageGenerationFeature.reduce(&state, .prepareExport(UUID()))
        #expect(state.export == ticket)
        let expectedStop = ImageGenerationTicket(owner: state.owner, sequence: state.sequence + 1)
        #expect(ImageGenerationFeature.reduce(&state, .ended) == [.cancelAll, .run(.stop(expectedStop), id: .stop)])
        _ = ImageGenerationFeature.reduce(&state, .exportFinished(ticket, true))
        #expect(state.export == nil && state.exportSucceeded && state.result != nil)
    }
}
