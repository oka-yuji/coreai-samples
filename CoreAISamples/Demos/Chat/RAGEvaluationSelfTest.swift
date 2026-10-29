import ConversationFeature
import ConversationRuntime
import Foundation
import FoundationModels
import Observation
import RAGContext

@MainActor
enum RAGEvaluationSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        do {
            let manifest = try fixture()
            try manifest.validate()
            guard let item = manifest.cases.first else { return }
            let replay = try RAGEvaluationReplay(manifest: manifest, item: item)
            var settings = RAGSettings()
            settings.web = true
            settings.today = manifest.settings.today
            let actual = await RAGRetriever.gather(
                question: item.question, settings: settings, count: { $0.count }, inputs: replay.inputs)
            let offsets = actual.sources.compactMap(\.offset)
            check(
                "rag-eval/flat-baseline", offsets == [0, 400, 800] && actual.pagesRead == 1,
                "offsets=\(offsets), pages=\(actual.pagesRead)")
            let missing = await RAGRetriever.gather(
                question: "unrecorded", settings: settings, count: { $0.count }, inputs: replay.inputs)
            check(
                "rag-eval/unknown-search",
                missing.sources.isEmpty && missing.notes.contains { $0.contains("not recorded") },
                "sources=\(missing.sources.count)")
            let unknown = await replay.execute(
                AgentToolCall(name: "fetch_page", arguments: [.init(name: "url", value: "https://unrecorded.example")]),
                settings: AgentToolSettings())
            check(
                "rag-eval/unknown-page", unknown.finalURL == nil && unknown.text.contains("not recorded"), unknown.text)
            await native(manifest: manifest, item: item, check: check)
        } catch { check("rag-eval/fixture", false, error.localizedDescription) }
    }

    private static func native(
        manifest: RAGEvaluationManifest, item: RAGEvaluationCase, check: (String, Bool, String) -> Void
    ) async {
        let directory = URL.temporaryDirectory.appending(path: "rag-eval-selftest-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            let replay = try RAGEvaluationReplay(manifest: manifest, item: item)
            let preferences = ConversationPreferences(persists: false)
            preferences.resetForHeadless()
            preferences.fixedDate = manifest.settings.today
            let runner = ConversationRunner(
                id: "rag-evaluation", chat: ChatViewModel(), preferences: preferences,
                archive: ConversationArchive(directory: directory),
                sessionFactory: { tools in
                    LanguageModelSession(model: RAGEvaluationFixtureModel(), tools: tools.tools(fileOperations: false))
                }, recordsMetrics: false,
                toolExecutor: { await replay.execute($0, settings: preferences.toolSettings) },
                sourceTokenCounter: { $0.count },
                retrieveSources: { question, settings, count in
                    await RAGRetriever.gather(
                        question: question, settings: settings, count: count, inputs: replay.inputs)
                })
            var initial = ConversationState(
                id: "rag-evaluation", now: Date(), mode: .work, settings: preferences.settings)
            initial.settings.webSearchEnabled = true
            let store = ConversationStore(initialState: initial, run: runner.run)
            store.send(.sendSources(item.question, .init(web: true, local: false, budget: 3_000), now: Date()))
            for await settled in Observations({ !store.state.isRunning && store.state.persistence == .saved })
            where settled { break }
            var result = RAGEvaluationResult(caseID: item.id, manifestHash: "fixture", strategy: "flat")
            RAGEvaluationCommand.collect(
                runner: runner, state: store.state, manifest: manifest, item: item, replay: replay,
                timedOut: false, answerTTFT: 1, result: &result)
            check(
                "rag-eval/work-metrics",
                result.status == .completed && result.outputTokens == 5
                    && result.inputTokens != nil && (result.sourceTokens ?? 0) > 0,
                "status=\(result.status) output=\(result.outputTokens ?? -1) sources=\(result.sourceTokens ?? -1)")
            var failed = store.state
            failed.persistence = .failed("fixture write failure")
            RAGEvaluationCommand.collect(
                runner: runner, state: failed, manifest: manifest, item: item, replay: replay,
                timedOut: false, answerTTFT: 1, result: &result)
            check(
                "rag-eval/failed-archive", result.status == .failed && result.reason?.contains("archive") == true,
                result.reason ?? "missing reason")
            let timing = RAGEvaluationProgress(start: .now)
            timing.observe(answerVisible: false, elapsed: 1)
            timing.observe(answerVisible: true, elapsed: 5)
            timing.observe(answerVisible: true, elapsed: 8)
            check("rag-eval/answer-timing", timing.firstAnswerSeconds == 5, "first=\(timing.firstAnswerSeconds ?? -1)")
            let placeholder = ConversationEntry(
                id: "missing", kind: .answer, title: "Answer", text: "(no answer text)", detail: .noAnswer)
            let preamble = ConversationEntry(id: "preamble", kind: .answer, title: "Answer", text: "I will check.")
            check(
                "rag-eval/no-answer-after-tool",
                RAGEvaluationCommand.answer([preamble, placeholder], guarder: CitationGuard(tools: runner.tools))
                    .isEmpty,
                "An earlier step is not the missing final answer")
            check(
                "rag-eval/no-answer-is-empty",
                RAGEvaluationCommand.answer([placeholder], guarder: runner.citationGuard).isEmpty,
                "UI placeholder is not model output")
            check(
                "rag-eval/native-work",
                store.state.lastFinish == .eos && store.state.mode == .work
                    && runner.lastRetrieval?.pagesRead == 1 && runner.tools.suppliedSourceIDs == ["S1"]
                    && runner.citationGuard.checks.last?.local?.violation == false
                    && replay.trace.withLock { $0.errors.isEmpty },
                "finish=\(store.state.lastFinish), supplied=\(runner.tools.suppliedSourceIDs)")
        } catch { check("rag-eval/native-work", false, error.localizedDescription) }
    }

    private static func fixture() throws -> RAGEvaluationManifest {
        let text = "Alpha interval is ten minutes.\n" + String(repeating: "Reference material. ", count: 55)
        let raw = "<p>" + text + "</p>"
        let normalized = AgentTools.plainText(fromHTML: raw)
        let hash = RAGContext.fingerprint(normalized)
        let object: [String: Any] = [
            "schemaVersion": 1, "datasetID": "native-fixture", "parserVersion": "flat-v1",
            "offsetUnit": "swift-character",
            "settings": [
                "modelID": "Qwen/Qwen3.8-27B", "today": "2026-10-29", "sourceTokenBudget": 3_000,
                "pageCharacterBudget": 3_000, "searchResults": 8, "pagesToRead": 5, "maxSteps": 8,
                "maxTokens": 1_536, "contextLimit": 16_384, "effort": "low", "citationGuard": false,
                "deadlineSeconds": 900,
            ],
            "documents": [
                [
                    "id": "alpha", "url": "https://rag.example/alpha", "title": "Alpha", "date": "", "format": "html",
                    "raw": raw, "text": normalized, "sha256": hash,
                ]
            ],
            "cases": [
                [
                    "id": "alpha-case", "split": "development", "category": "heading", "language": "en",
                    "question": "Alpha interval", "documentIDs": ["alpha"],
                    "gold": [
                        ["documentID": "alpha", "version": hash, "offset": 0, "text": "Alpha interval is ten minutes."]
                    ],
                    "expectedPoints": ["ten minutes"], "expectsAbstention": false,
                ]
            ],
        ]
        return try JSONDecoder().decode(
            RAGEvaluationManifest.self, from: JSONSerialization.data(withJSONObject: object))
    }
}

private nonisolated struct RAGEvaluationFixtureModel: LanguageModel {
    struct Executor: LanguageModelExecutor {
        typealias Model = RAGEvaluationFixtureModel
        init(configuration: UUID) {}

        func respond(
            to request: LanguageModelExecutorGenerationRequest, model: Model,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            try Task.checkCancellation()
            await channel.send(.response(action: .appendText("Ten minutes [S1].", tokenCount: 5)))
            await channel.send(.response(action: .updateMetadata(["stopReason": "stopToken"])))
        }
    }

    let executorConfiguration = UUID()
    let capabilities = LanguageModelCapabilities([.reasoning, .toolCalling])
}
