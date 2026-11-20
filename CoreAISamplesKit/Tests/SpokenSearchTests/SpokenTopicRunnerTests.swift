import Foundation
import SpokenSearchFeature
import Testing

@testable import SpokenSearchRuntime

@MainActor
struct SpokenTopicRunnerTests {
    @Test("A model lease arriving after a variant switch is returned before any embedding is loaded")
    func lateLease() async {
        let speech = SuspendedTranscription()
        let model = SpokenModelProbe()
        let runner = SpokenSearchRunner(
            isPermitted: { true }, speech: speech,
            makeTopicIndex: { url in SpokenTopicIndex(folder: url, makeModel: { model }) })
        let folder = URL(filePath: "/models")
        let old = SpokenTopicLeaseProbe(delayed: true)
        let new = SpokenTopicLeaseProbe(delayed: false)
        runner.setTopicSource(SpokenTopicSource(identity: "old", directory: folder) { await old.acquire() })
        let open = Task { await runner.run(.open(1, URL(filePath: "/memo.wav"))) { _ in } }
        await speech.waitForStart()
        speech.finish()
        await open.value
        var oldCallbacks = 0
        let query = Task { await runner.run(.findTopic(1, "coat", "old")) { _ in oldCallbacks += 1 } }
        await old.waitForRequest()
        runner.setTopicSource(
            SpokenTopicSource(identity: "new", directory: folder, validate: { new.isRegistered }) {
                await new.acquire()
            })
        old.resume()
        await query.value
        #expect(old.acquisitions == 1 && old.releases == 1 && oldCallbacks == 0)
        #expect(await model.loads == 0)
        var result: [SpokenHit] = []
        await runner.run(.findTopic(1, "coat", "new")) { action in
            if case .topicFound(1, "new", let hits) = action { result = hits }
        }
        #expect(result.map(\.text) == ["day"])
        #expect(new.acquisitions == 1 && new.releases == 1)
        #expect(await model.unloads == 1)
        await runner.run(.waitForTopic(1, discardIndex: true)) { _ in }
        await runner.run(.findTopic(1, "coat", "new")) { _ in }
        #expect(await model.documents == ["day", "day"])
        new.isRegistered = false
        var expired = false
        if let hit = result.first {
            await runner.run(.play(1, hit, topicIdentity: "new")) {
                if case .topicExpired(1, "new") = $0 { expired = true }
            }
        }
        #expect(expired && new.acquisitions == 2 && new.releases == 2)
        await runner.close()
    }
}
