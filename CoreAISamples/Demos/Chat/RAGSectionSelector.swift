@_spi(Verification) import CoreAIBackend
import Foundation
import FoundationModels
import Observation
import RAGContext

@MainActor
final class RAGSectionSelector {
    private enum Event: Sendable {
        case finished(String?)
        case timeout
        case timerCancelled
    }

    let makeSession: (String) -> LanguageModelSession
    let countInput: @Sendable (String, String) async throws -> Int
    let wait: @Sendable (Duration) async throws -> Void
    let stopping: @Sendable () -> Void
    let finishExecution: @Sendable () async -> Void
    let measurement: @Sendable () async -> GenerationMeasurement?

    init(
        makeSession: @escaping (String) -> LanguageModelSession,
        countInput: @escaping @Sendable (String, String) async throws -> Int,
        wait: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) },
        stopping: @escaping @Sendable () -> Void = {},
        finishExecution: @escaping @Sendable () async -> Void = {},
        measurement: @escaping @Sendable () async -> GenerationMeasurement? = { nil }
    ) {
        self.makeSession = makeSession
        self.countInput = countInput
        self.wait = wait
        self.stopping = stopping
        self.finishExecution = finishExecution
        self.measurement = measurement
    }

    convenience init(model: LocalLanguageModel) {
        let measured = model.recordingMeasurements(.summary)
        self.init(
            makeSession: { LanguageModelSession(model: measured, tools: [], instructions: $0) },
            countInput: { try await model.countToolFreePromptTokens(instructions: $0, prompt: $1) },
            finishExecution: { _ = await model.processedTokens() },
            measurement: { await model.lastRequestMeasurement() })
    }

    func select(
        _ baseline: RAGRetrieval, question: String, budget: Int, count: @Sendable (String) -> Int,
        contextLimit: Int, deadline: Date?
    ) async -> RAGRetrieval {
        var result = baseline
        var trace = RAGSectionTrace()
        let started = ContinuousClock.now
        do {
            try await choose(
                &result, trace: &trace, question: question, budget: budget, count: count,
                contextLimit: contextLimit, deadline: deadline)
        } catch {
            if Task.isCancelled || error is CancellationError {
                trace.status = "cancelled"
                trace.reason = "cancelled"
            } else {
                trace.status = "failed"
                trace.reason = error.localizedDescription
                result.selectionFailure = error.localizedDescription
            }
        }
        trace.seconds = (ContinuousClock.now - started) / .seconds(1)
        result.selectionTrace = trace
        if trace.status == "fallback", let reason = trace.reason { result.fallbacks.append("selection: " + reason) }
        return result
    }

    private func choose(
        _ result: inout RAGRetrieval, trace: inout RAGSectionTrace, question: String, budget: Int,
        count: @Sendable (String) -> Int, contextLimit: Int, deadline: Date?
    ) async throws {
        try Task.checkCancellation()
        let limits = result.selectionLimits
        guard limits.isValid else { throw RAGEvaluationManifest.Failure.invalid("selection limits") }
        let originalPool = result.selectionCandidates
        var items = RAGSectionSelection.menu(
            originalPool, question: question, limit: limits.candidates,
            rank: RAGRetriever.bestRanking)
        trace.status = "fallback"
        trace.reason = items.isEmpty ? "noCandidates" : "inputBudgetExceeded"
        let instructions = RAGSectionSelection.instructions(limit: limits.sections)
        var prompt = ""
        while !items.isEmpty {
            try Task.checkCancellation()
            prompt = try RAGSectionSelection.prompt(question: question, items: items)
            let tokens = try await countInput(instructions, prompt)
            trace.preparedInputTokens = tokens
            if tokens > 0 && tokens <= min(limits.inputTokens, contextLimit - limits.outputTokens) { break }
            items.removeLast()
        }
        trace.candidateIDs = items.map(\.sectionID)
        let eligible = Set(trace.candidateIDs)
        let pool = result.selectionCandidates.filter { eligible.contains($0.sectionID) }
        result.selectionCandidates = pool
        guard !items.isEmpty else { return }
        let remaining = deadline?.timeIntervalSinceNow ?? limits.timeoutSeconds
        guard remaining > 0 else {
            trace.status = "cancelled"
            trace.reason = "deadline"
            return
        }
        try Task.checkCancellation()
        let previousMeasurementID = await measurement()?.requestID
        try Task.checkCancellation()
        trace.didRun = true
        let native = makeSession(instructions)
        native.transcriptErrorHandlingPolicy = .preserveTranscript
        let options = GenerationOptions(
            samplingMode: .greedy, maximumResponseTokens: limits.outputTokens,
            toolCallingMode: .disallowed)
        let context = ContextOptions(reasoningLevel: .custom("off"))
        let sleeper = wait
        let timeout = min(limits.timeoutSeconds, remaining)
        let request = prompt
        var stoppedAt: ContinuousClock.Instant?
        let event = await withTaskGroup(of: Event.self) { group in
            group.addTask {
                do {
                    try Task.checkCancellation()
                    _ = try await native.respond(to: request, options: options, contextOptions: context)
                    return .finished(nil)
                } catch { return .finished(error.localizedDescription) }
            }
            group.addTask {
                do {
                    try await sleeper(.seconds(timeout))
                    return .timeout
                } catch { return .timerCancelled }
            }
            let first = await group.next() ?? .timerCancelled
            if case .timeout = first { stoppedAt = .now }
            if Task.isCancelled { stoppedAt = .now }
            if stoppedAt != nil { stopping() }
            group.cancelAll()
            return first
        }
        await withTaskCancellationShield {
            for await done in Observations({ !native.isResponding }) where done { break }
            await finishExecution()
        }
        if let stoppedAt { trace.stopSeconds = (ContinuousClock.now - stoppedAt) / .seconds(1) }
        let latest = await measurement()
        let measured = latest?.requestID == previousMeasurementID ? nil : latest
        trace.prefilledTokens = measured?.prefilledTokens
        trace.sampledTokens = measured?.sampledTokens
        trace.nativeOutcome = measured?.outcome?.rawValue
        if case .finished(nil) = event, !Task.isCancelled {
            trace.usageStatus = "reported"
            trace.inputTokens = native.usage.input.totalTokenCount
            trace.outputTokens = native.usage.output.totalTokenCount
            trace.reusedTokens = native.usage.input.cachedTokenCount
        } else if let measured {
            trace.usageStatus = "runtimePartial"
            trace.inputTokens = measured.promptTokens
            trace.outputTokens = measured.sampledTokens
        } else {
            trace.usageStatus = "unavailable"
        }
        trace.response = ConversationSession.responseText(Array(native.transcript)[...])
        try Task.checkCancellation()
        if let deadline, deadline <= Date() {
            trace.status = "cancelled"
            trace.reason = "deadline"
            return
        }
        switch event {
        case .timeout:
            trace.reason = "timeout"
            return
        case .timerCancelled: throw CancellationError()
        case .finished(let error):
            if let error {
                trace.status = "failed"
                trace.reason = error
                result.selectionFailure = error
                return
            }
        }
        guard (trace.outputTokens ?? 0) <= limits.outputTokens else {
            trace.reason = "outputBudgetExceeded"
            return
        }
        let choice = RAGSectionSelection.parse(trace.response, items: items, limit: limits.sections)
        switch choice {
        case .invalid(let reason):
            trace.reason = reason
            return
        case .none:
            trace.status = "none"
            trace.selectedIDs = []
            result.sources = []
            result.bodySources = []
            result.atomicGroups = []
        case .sections(let ids):
            trace.status = "selected"
            trace.selectedIDs = ids
            let selected = RAGSectionSelection.bodies(
                for: ids, candidates: originalPool, question: question, rank: RAGRetriever.bestRanking)
            let packed = RAGStructuredSelection.select(
                selected, question: question, budget: budget, count: count,
                rank: { values, _ in Array(values.indices) })
            result.sources = packed.sources
            result.bodySources = packed.candidates.map(\.source)
            result.atomicGroups = packed.candidates.map { $0.headers + [$0.source] }
        }
        trace.reason = nil
        result.effectiveStrategy = RAGStrategy.structureLLM.rawValue
        result.sourceTokens = result.sources.reduce(0) { $0 + count(RAGContext.measuredBlock($1)) }
    }
}
