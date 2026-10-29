import ConversationFeature
import ConversationRuntime
import CoreAIBackend
import Darwin
import Foundation
import Observation
import RAGContext

@MainActor
enum RAGEvaluationCommand {
    static func run(manifestURL: URL) async -> Int32 {
        do {
            let data = try Data(contentsOf: manifestURL)
            let manifest = try JSONDecoder().decode(RAGEvaluationManifest.self, from: data)
            let strategy = LaunchArguments.value("--rag-strategy") ?? "flat"
            try manifest.validate(strategy: strategy, requiresModelPin: !LaunchArguments.flag("--validate-only"))
            for item in manifest.cases { _ = try RAGEvaluationReplay(manifest: manifest, item: item) }
            if LaunchArguments.flag("--validate-only") {
                if let output = LaunchArguments.value("--structure-output") {
                    try await RAGStructureInspection.inspect(manifest, output: URL(filePath: output))
                }
                print("RAG EVALUATION VALID: \(manifest.cases.count) cases, \(manifest.documents.count) documents")
                return 0
            }
            guard let id = LaunchArguments.value("--case"),
                let item = manifest.cases.first(where: { $0.id == id }),
                let output = LaunchArguments.value("--output"),
                let model = LaunchArguments.value("--model"),
                let mode = ConversationMode(rawValue: LaunchArguments.value("--mode") ?? "work")
            else {
                throw RAGEvaluationManifest.Failure.invalid("requires --case, --output, --model and a valid --mode")
            }
            let directory = URL(filePath: output, directoryHint: .isDirectory)
            guard !FileManager.default.fileExists(atPath: directory.path) else {
                throw RAGEvaluationManifest.Failure.invalid("output already exists")
            }
            try FileManager.default.createDirectory(
                at: directory.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard mkdir(directory.path, 0o700) == 0 else {
                throw RAGEvaluationManifest.Failure.invalid("could not create a new output directory")
            }
            try data.write(to: directory.appending(path: "manifest.json"), options: .atomic)
            var result = RAGEvaluationResult(
                caseID: id, manifestHash: RAGContext.fingerprint(data),
                strategy: strategy)
            result.mode = mode.rawValue
            try write(result, to: directory)
            do {
                result.status = .running
                result.osVersion = ProcessInfo.processInfo.operatingSystemVersionString
                try write(result, to: directory)
                try await execute(
                    manifest: manifest, item: item, mode: mode, model: URL(filePath: model),
                    directory: directory, result: &result)
            } catch {
                result.status = error is CancellationError ? .cancelled : .failed
                result.reason = error.localizedDescription
            }
            result.lifetimePeakFootprintBytes = peakFootprint()
            result.memoryStatus = result.lifetimePeakFootprintBytes == nil ? "unavailable" : "kernelLifetimePeak"
            try write(result, to: directory)
            print("RAG EVALUATION \(result.status.rawValue): \(id)")
            return result.status == .completed ? 0 : 1
        } catch {
            FileHandle.standardError.write(Data(("RAG EVALUATION ERROR: " + error.localizedDescription + "\n").utf8))
            return 1
        }
    }

    private static func execute(
        manifest: RAGEvaluationManifest, item: RAGEvaluationCase, mode: ConversationMode, model: URL,
        directory: URL, result: inout RAGEvaluationResult
    ) async throws {
        let bundle = try ModelBundle(url: model)
        guard bundle.sourceModelID == manifest.settings.modelID,
            let hash = bundle.mainHash, hash == manifest.settings.expectedModelHash
        else {
            throw RAGEvaluationManifest.Failure.invalid("requires the pinned Qwen3.8 model")
        }
        result.modelID = bundle.sourceModelID
        result.modelHash = bundle.mainHash
        let tokenizer = try String(contentsOf: bundle.tokenizerURL.appending(path: "tokenizer.json"), encoding: .utf8)
        result.tokenizerHash = RAGContext.fingerprint(tokenizer)
        let chat = ChatViewModel()
        guard chat.contextLimit == manifest.settings.contextLimit else {
            throw RAGEvaluationManifest.Failure.invalid("context limit differs from manifest")
        }
        await chat.load(model)
        guard chat.phase == .ready else { throw RAGEvaluationManifest.Failure.invalid("model load failed") }
        let preferences = ConversationPreferences(persists: false)
        preferences.resetForHeadless()
        preferences.effort = AgentReasoningEffort(rawValue: manifest.settings.effort) ?? .low
        preferences.maxSteps = manifest.settings.maxSteps
        preferences.maxTokens = manifest.settings.maxTokens
        preferences.pageBudget = manifest.settings.pageCharacterBudget
        preferences.citationGuard = manifest.settings.citationGuard
        preferences.fixedDate = manifest.settings.today
        preferences.stableNotePaths = true
        preferences.notesFolderPath = directory.appending(path: "notes").path
        let replay = try RAGEvaluationReplay(manifest: manifest, item: item)
        let inputs = replay.inputs
        let configured = RAGInputs(
            configure: { original in
                var settings = original
                settings.searchResults = manifest.settings.searchResults
                settings.pagesToRead = manifest.settings.pagesToRead
                settings.pageCharacterBudget = manifest.settings.pageCharacterBudget
                settings.today = manifest.settings.today
                settings.recordsDiagnostics = true
                return settings
            },
            local: inputs.local,
            search: { query, original in
                var settings = original
                settings.searchResults = manifest.settings.searchResults
                return try await inputs.search(query, settings)
            }, read: inputs.read)
        let host = ConversationHost(
            chat: chat, preferences: preferences,
            archive: ConversationArchive(directory: directory.appending(path: "conversations")),
            retrievalInputs: configured,
            toolExecutor: { await replay.execute($0, settings: preferences.toolSettings) })
        chat.ragEnabled = true
        chat.ragWeb = replay.documents.contains(where: \.isWeb)
        chat.ragLocal = replay.documents.contains { !$0.isWeb }
        chat.ragBudget = manifest.settings.sourceTokenBudget
        let start = ContinuousClock.now
        preferences.deadline = Date().addingTimeInterval(manifest.settings.deadlineSeconds)
        guard host.send(item.question, mode: mode) else {
            throw RAGEvaluationManifest.Failure.invalid("conversation refused the request")
        }
        let progress = await wait(host, deadline: manifest.settings.deadlineSeconds, start: start)
        result.totalSeconds = (ContinuousClock.now - start) / .seconds(1)
        collect(
            runner: host.runner, state: host.state, manifest: manifest, item: item,
            replay: replay, timedOut: progress.expired, answerTTFT: progress.firstAnswerSeconds, result: &result)
    }

    static func collect(
        runner: ConversationRunner, state: ConversationState, manifest: RAGEvaluationManifest,
        item: RAGEvaluationCase, replay: RAGEvaluationReplay, timedOut: Bool, answerTTFT: Double?,
        result: inout RAGEvaluationResult
    ) {
        result.effectiveStrategy = "flat"
        let usage = runner.session.usage
        result.answer = answer(state.entries, guarder: runner.citationGuard)
        result.inputTokens = usage?.input.totalTokenCount
        result.reusedTokens = usage?.input.cachedTokenCount
        result.outputTokens = usage?.output.totalTokenCount
        result.sourceTokens = runner.lastRetrieval?.sourceTokens
        result.retrievalSeconds = runner.lastRetrieval?.seconds
        result.selectionSeconds = 0
        result.answerTTFTSeconds = answerTTFT
        result.sourceIDsValid = runner.citationGuard.checks.last?.local.map { !$0.violation }
        result.finishReason = state.lastFinish.rawValue
        result.citationOutcome = runner.citationGuard.outcome
        result.documents = runner.lastRetrieval?.documents
        result.candidates = (runner.lastRetrieval?.candidatePassages ?? []).compactMap(passage)
        let supplied = runner.tools.sources.records.filter {
            runner.tools.suppliedSourceIDs.contains($0.id)
        }
        result.suppliedPassages = (runner.lastRetrieval?.sources ?? []).filter { source in
            supplied.contains { $0.document == source.document && $0.version == source.version }
        }.compactMap(passage)
        result.retrievalMetrics = .measure(item: item, manifest: manifest, passages: result.suppliedPassages)
        let trace = replay.trace.withLock { $0 }
        result.toolRequests = trace.requests
        result.replayErrors = trace.errors
        if timedOut {
            result.status = .cancelled
            result.reason = "deadline"
        } else if state.persistence != .saved {
            result.status = .failed
            if case .failed(let message) = state.persistence {
                result.reason = "archive: " + message
            } else {
                result.reason = "archive was not saved"
            }
        } else if state.lastFinish == .eos || state.lastFinish == .cap {
            result.status = trace.errors.isEmpty && usage != nil ? .completed : .failed
            result.reason =
                !trace.errors.isEmpty ? "snapshot request not recorded" : usage == nil ? "usage unavailable" : nil
        } else {
            result.status = .failed
            result.reason = state.turns.last?.failure ?? state.lastFinish.rawValue
        }
    }

    static func answer(_ entries: [ConversationEntry], guarder: CitationGuard) -> String {
        if let answer = guarder.guardedAnswer { return answer }
        guard let last = entries.last(where: { $0.kind == .answer }), last.detail != .noAnswer else { return "" }
        return last.text
    }

    private static func wait(
        _ host: ConversationHost, deadline: Double, start: ContinuousClock.Instant
    ) async -> RAGEvaluationProgress {
        let state = RAGEvaluationProgress(start: start)
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await settled(host, state: state) }
            group.addTask { await stop(host, after: deadline, state: state) }
            await group.next()
            group.cancelAll()
        }
        return state
    }

    private static func settled(_ host: ConversationHost, state: RAGEvaluationProgress) async {
        for await observation in Observations({
            (
                host.state.isRunning || host.state.isSaving || host.chat.externallyBusy,
                host.state.entries.contains { $0.kind == .answer && $0.detail != .noAnswer && !$0.text.isEmpty }
            )
        }) {
            state.observe(answerVisible: observation.1, elapsed: (ContinuousClock.now - state.start) / .seconds(1))
            if !observation.0 { break }
        }
    }

    private static func stop(_ host: ConversationHost, after seconds: Double, state: RAGEvaluationProgress) async {
        do { try await Task.sleep(for: .seconds(seconds)) } catch { return }
        state.expired = true
        host.stop()
        await settled(host, state: state)
    }

    private static func passage(_ source: RAGSource) -> RAGEvaluationPassage? {
        guard let document = source.document, let version = source.version, let offset = source.offset else {
            return nil
        }
        return RAGEvaluationPassage(document: document, version: version, offset: offset, text: source.text)
    }

    private static func write(_ result: RAGEvaluationResult, to directory: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(result).write(to: directory.appending(path: "result.json"), options: .atomic)
    }

    private static func peakFootprint() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? UInt64(max(0, info.ledger_phys_footprint_peak)) : nil
    }
}

@MainActor
final class RAGEvaluationProgress {
    let start: ContinuousClock.Instant
    var expired = false
    private(set) var firstAnswerSeconds: Double?

    init(start: ContinuousClock.Instant) { self.start = start }

    func observe(answerVisible: Bool, elapsed: Double) {
        if answerVisible, firstAnswerSeconds == nil { firstAnswerSeconds = elapsed }
    }
}
