import CoreAI
import Foundation
import FoundationModels
import Testing

@_spi(Verification) @testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1)))
struct GenerationMeasurementTests {
    enum Scenario: String, CaseIterable, Sendable {
        case stopToken
        case maxTokens
        case cancelled
        case stepFailure
        case readFailure

        var outcome: GenerationMeasurement.Outcome {
            switch self {
            case .stopToken: .stopToken
            case .maxTokens: .maxTokens
            case .cancelled: .cancelled
            case .stepFailure, .readFailure: .failed
            }
        }

        var counts: (sampled: Int, steps: Int, sends: Int, rows: Int) {
            switch self {
            case .stopToken: (1, 1, 1, 2)
            case .maxTokens: (8, 7, 8, 8)
            case .cancelled, .stepFailure: (1, 0, 1, 1)
            case .readFailure: (0, 0, 0, 0)
            }
        }

        var error: String? {
            switch self {
            case .stopToken, .maxTokens: nil
            case .cancelled: String(describing: CancellationError())
            case .stepFailure: String(describing: ReloadTestFailure.step)
            case .readFailure: String(describing: ReloadTestFailure.read)
            }
        }
    }

    struct Behavior: Equatable {
        var text: String
        var generated: [Int32]
        var processed: [Int32]
        var stopReason: String?
        var error: String?
    }

    struct Run {
        var behavior: Behavior
        var requestID: UUID?
        var measurement: GenerationMeasurement?
    }

    static func run(_ scenario: Scenario, level: GenerationMeasurement.Level) async throws -> Run {
        var settings = LocalLanguageModel.VerificationSettings()
        settings.measurement = level
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0), settings: settings)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        fixture.control.state.withLock { state in
            switch scenario {
            case .stopToken: break
            case .maxTokens: state.generationTokens = [1]
            case .cancelled: state.blockNextStep = true
            case .stepFailure: state.failNextStep = true
            case .readFailure: state.failNextRead = true
            }
        }
        var updates = fixture.model.preparationUpdates().makeAsyncIterator()
        let session = LanguageModelSession(model: fixture.model)
        let (cancelledText, error) = await withTaskCancellationHandler {
            let consumer = Task { () -> String? in
                do {
                    for try await _ in session.streamResponse(
                        to: "a", options: .init(samplingMode: .greedy, maximumResponseTokens: 8))
                    {}
                    return nil
                } catch {
                    return String(reflecting: type(of: error))
                }
            }
            var text: String?
            if scenario == .cancelled {
                await waitUntil { !responseText(in: session.transcript, after: 0).isEmpty }
                text = responseText(in: session.transcript, after: 0)
                consumer.cancel()
            }
            let error = await consumer.value
            _ = await fixture.model.processedTokens()
            fixture.control.resumeStep()
            await waitUntil { !session.isResponding }
            return (text, error)
        } onCancel: {
            fixture.control.resumeStep()
        }
        let behavior = Behavior(
            text: cancelledText ?? responseText(in: session.transcript, after: 0),
            generated: await fixture.model.lastGeneratedTokens(), processed: await fixture.model.processedTokens(),
            stopReason: stopReason(in: session.transcript), error: error)
        return Run(
            behavior: behavior, requestID: await updates.next()?.requestID,
            measurement: await fixture.model.lastRequestMeasurement())
    }

    static func stopReason(in transcript: Transcript) -> String? {
        let response = transcript.reversed().lazy.compactMap { entry -> Transcript.Response? in
            if case .response(let response) = entry { return response }
            return nil
        }.first
        return response?.metadata["stopReason"].flatMap { try? $0.value(String.self) }
    }

    @Test(
        "The same fixture records the actual counts and time intervals of the EOS, limit, cancellation and failure ends",
        arguments: Scenario.allCases, [GenerationMeasurement.Level.summary, .trace])
    func recordsCountsAndIntervals(_ scenario: Scenario, level: GenerationMeasurement.Level) async throws {
        let run = try await Self.run(scenario, level: level)
        let measurement = try #require(run.measurement)
        let counts = scenario.counts
        #expect(measurement.requestID == run.requestID)
        #expect(measurement.level == level)
        #expect(measurement.outcome == scenario.outcome)
        #expect(measurement.error == scenario.error)
        #expect(measurement.promptTokens == 1)
        #expect(measurement.prefilledTokens == (scenario == .readFailure ? 0 : 1))
        #expect(measurement.sampledTokens == counts.sampled)
        #expect(measurement.steps == counts.steps)
        #expect(measurement.sends == counts.sends)
        #expect(measurement.trace.count == (level == .trace ? counts.rows : 0))
        #expect(measurement.trace.filter { $0.stepSeconds != nil }.count == (level == .trace ? counts.steps : 0))
        #expect(measurement.trace.filter { $0.sendSeconds != nil }.count == (level == .trace ? counts.sampled : 0))
        #expect((measurement.firstTokenSeconds == nil) == (counts.rows == 0))
        #expect((measurement.firstSendSeconds == nil) == (counts.sends == 0))
        if let first = measurement.firstTokenSeconds, let last = measurement.lastTokenSeconds {
            #expect(measurement.readSeconds <= first && first <= last && last <= measurement.totalSeconds)
        }
        let stages = [
            measurement.readSeconds, measurement.sampleSeconds, measurement.textSeconds, measurement.sendSeconds,
            measurement.stepSeconds, measurement.drainSeconds,
        ]
        #expect(stages.allSatisfy { $0 >= 0 })
        #expect(stages.reduce(0, +) <= measurement.totalSeconds + 1e-9)
    }

    @Test(
        "Measurement does not change the generated tokens or the stop and error behavior", arguments: Scenario.allCases)
    func measurementKeepsBehavior(_ scenario: Scenario) async throws {
        let plain = try await Self.run(scenario, level: .off)
        let traced = try await Self.run(scenario, level: .trace)
        #expect(plain.measurement == nil)
        #expect(traced.measurement?.outcome == scenario.outcome)
        #expect(plain.behavior == traced.behavior)
        #expect(plain.behavior.generated == Array(repeating: 1, count: scenario.counts.sampled))
        #expect(
            plain.behavior.stopReason
                == ([.stopToken, .maxTokens].contains(scenario.outcome) ? scenario.outcome.rawValue : nil))
    }

    @Test("A request rejected by request-only validation keeps the previous measurement")
    func rejectionKeepsMeasurement() async throws {
        var settings = LocalLanguageModel.VerificationSettings()
        settings.measurement = .summary
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0), settings: settings)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        _ = try await fixture.request()
        let previous = try #require(await fixture.model.lastRequestMeasurement())
        let invalid = LanguageModelSession(model: fixture.model)
        await #expect(throws: LanguageModelError.self) {
            try await invalid.respond(to: "a", contextOptions: .init(reasoningLevel: .custom("unsupported")))
        }
        #expect(await fixture.model.lastRequestMeasurement() == previous)
    }

    @Test(
        "The load statistics name the requested compute unit and report default without a preference",
        arguments: [(ComputeUnitKind?.none, "default"), (.cpu, "cpu"), (.gpu, "gpu"), (.neuralEngine, "neuralEngine")])
    func computeUnitNames(kind: ComputeUnitKind?, name: String) {
        #expect(Engine.computeUnitName(kind) == name)
    }
}
