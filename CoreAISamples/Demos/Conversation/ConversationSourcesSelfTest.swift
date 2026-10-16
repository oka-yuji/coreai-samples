import ConversationFeature
import Foundation
import RAGContext
import Synchronization

@MainActor
enum ConversationSourcesSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        let directory = URL.temporaryDirectory.appending(path: "sources-selftest-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data("Alpha is the first stage.".utf8).write(to: directory.appending(path: "manual.md"))
            var settings = RAGSettings()
            settings.local = true
            settings.documentsDirectory = directory
            let threadChecks = Mutex<[Bool]>([])
            let retrieval = await RAGRetriever.gather(
                question: "What is Alpha?", settings: settings,
                count: {
                    threadChecks.withLock { $0.append(Thread.isMainThread) }
                    return $0.count
                })
            check(
                "sources/retrievalOffMainThread", threadChecks.withLock { !$0.isEmpty && !$0.contains(true) },
                "mainThread=\(threadChecks.withLock { $0 })")
            let preferences = ConversationPreferences(persists: false)
            let runner = ConversationRunner(
                id: "source-check", chat: ChatViewModel(), preferences: preferences)
            let ids = try runner.tools.registerSources(retrieval.conversationSources())
            check(
                "sources/collectedDoesNotMeanSupplied",
                runner.tools.sources.check(answer: "Alpha [S1]", suppliedIDs: runner.tools.suppliedSourceIDs).violation,
                "supplied=\(runner.tools.suppliedSourceIDs)")
            runner.tools.markSourcesSupplied(ids)
            let web = ConversationSource(
                document: "https://example.com/manual", version: String(repeating: "b", count: 64), title: "Web",
                passages: [SourcePassage(offset: 0, text: "Alpha"), SourcePassage(offset: 500, text: "Beta")])
            _ = try runner.tools.registerSources([web])
            check(
                "sources/webLedgerUsesSelectedBody",
                runner.tools.ledger.count == 1
                    && runner.tools.ledger[0].sha256 == AgentSelfTest.sha256("Alpha\n\nBeta")
                    && runner.tools.ledger[0].characters == "Alpha\n\nBeta".count
                    && runner.tools.seenPages[web.document] == "Alpha\n\nBeta", "records=\(runner.tools.ledger.count)")
            runner.citationGuard.begin(active: false, suppliedSourcesRequired: true)
            _ = await runner.citationGuard.settle(
                TurnEnd(kind: .ended(stopReason: "stopToken"), outcome: TurnEnd.answered, answer: "Alpha [S1]."),
                step: 1, maxSteps: 1, stopped: false, usage: nil)
            check(
                "sources/localRetrievalAndCitation",
                ids == ["S1"]
                    && runner.tools.sources.records[0].passages[0].text.contains("Alpha")
                    && runner.citationGuard.checks.last?.local?.violation == false,
                "ids=\(ids)")
            runner.citationGuard.begin(active: false, suppliedSourcesRequired: true)
            _ = await runner.citationGuard.settle(
                TurnEnd(kind: .ended(stopReason: "stopToken"), outcome: TurnEnd.answered, answer: "Alpha [S99]."),
                step: 1, maxSteps: 1, stopped: false, usage: nil)
            check(
                "sources/unknownIDsHeldWithoutWebGuard", runner.citationGuard.outcome == "held",
                runner.citationGuard.reason)
            let (started, signal) = AsyncStream<Void>.makeStream()
            let (finished, completion) = AsyncStream<Bool>.makeStream()
            runner.session.prepareRun(10)
            let cancelled = await withTaskGroup(of: Bool.self) { group in
                group.addTask {
                    let result = await runner.collectSources(run: 10) {
                        signal.yield(())
                        try? await Task.sleep(for: .seconds(30))
                        completion.yield(Task.isCancelled)
                        completion.finish()
                        return RAGRetrieval()
                    }
                    return result == nil
                }
                group.addTask {
                    for await _ in started { break }
                    await runner.run(.stop(run: 10)) { _ in }
                    for await cancelled in finished { return cancelled }
                    return false
                }
                var pass = true
                for await result in group { pass = pass && result }
                return pass
            }
            signal.finish()
            check("sources/stopJoinsRetrieval", cancelled, "cancelled=\(cancelled)")
            var failed = ConversationState(id: "failed-followup", now: Date())
            _ = ConversationFeature.reduce(&failed, .send("Follow up", mode: .chat, now: Date()))
            _ = ConversationFeature.reduce(&failed, .generationFailed(run: 1, message: "Context full"))
            let previous = GenerationStats(answer: "Earlier [S1].", outputTokens: 10, stopReason: "stopToken")
            check(
                "sources/guiRejectsPreviousAnswer",
                !GUIDriver.ragReplyPassed(
                    previous, turn: failed.turns.last, check: runner.tools.sources.check(answer: previous.answer)),
                "currentFailure=\(failed.turns.last?.failure ?? "-")")
        } catch {
            check("sources/integration", false, error.localizedDescription)
        }
    }
}
