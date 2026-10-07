import DecideFeature
import DecideRuntime
import FeatureStore
import Foundation

@MainActor
enum DecideSelfTest {
    static func run(directory: String) async {
        do {
            try await perform(directory: URL(filePath: directory))
        } catch {
            FileHandle.standardError.write(Data(("Decide scenario failed: \(error.localizedDescription)\n").utf8))
            exit(1)
        }
    }

    private static func perform(directory: URL) async throws {
        guard
            let path = LaunchArguments.value("--decision-bundle")
                ?? ProcessInfo.processInfo.environment["COREAI_DECISION_BUNDLE"]
        else { throw DecideScenarioError("Provide --decision-bundle or COREAI_DECISION_BUNDLE.") }
        let service = LiveDecideService(
            examplesURL: Bundle.main.url(forResource: "decide-examples", withExtension: "json"))
        let harness = DecideScenarioHarness(service: service)
        let store = harness.store
        let files = DecideFiles()
        store.send(.appeared)
        await harness.settle()
        guard let example = store.state.examples.first(where: { $0.id == "contract" }) else {
            throw DecideScenarioError("The bundled contract example is missing.")
        }
        store.send(.replaceInput(example.input))
        store.send(.openBundle(path))
        await harness.settle()
        try require(store.state.model != nil && store.state.canRun, store.state.message ?? "The model is not ready.")
        try require(store.state.input.items.count == 4, "The contract must start with four items.")
        store.send(.run(reuse: true))
        await harness.settle()
        guard let initial = store.state.output else {
            throw DecideScenarioError(store.state.message ?? "No initial result.")
        }
        try require(initial.rows.count == 4 && !initial.reusedDocument, "The initial document was not evaluated.")
        let firstReview = initial.rows.first { $0.reviewFirst } ?? initial.rows[0]
        store.send(.confirm(firstReview.id, firstReview.candidate))
        try require(store.state.confirmations.count == 1, "Reviewer confirmation was not stored.")
        store.send(.addItem)
        store.send(.updateItem(example.extraItem))
        await harness.settle()
        try require(store.state.confirmations.isEmpty, "Editing retained a stale reviewer confirmation.")
        store.send(.run(reuse: true))
        await harness.settle()
        guard let reused = store.state.output else { throw DecideScenarioError("No reuse result.") }
        try require(
            reused.rows.count == 5 && reused.reusedDocument && reused.reusedTokens > 0, "The document was not reused.")
        try require(
            reused.recomputedAll && store.state.changedItems.contains(example.extraItem.id), "Changes were not marked.")
        try require(store.state.input.document.contains(example.searchHint), "The source search hint was not found.")
        store.send(.setLowestFirst(true))
        let probabilities = store.state.rows.map(\.maximumProbability)
        try require(probabilities == probabilities.sorted(), "Review ordering is incorrect.")
        let review = reused.rows.first { $0.reviewFirst } ?? reused.rows[0]
        store.send(.confirm(review.id, review.candidate))
        store.send(.prepareExport)
        await harness.waitForExport()
        guard let reusedExport = store.state.exportData else {
            throw DecideScenarioError("The reuse export is missing.")
        }
        try await files.write(reusedExport, to: directory.appendingPathComponent("review-reused.json"))
        if let ticket = store.state.exportTicket { store.send(.dismissExport(ticket)) }
        store.send(.run(reuse: false))
        await harness.settle()
        guard let fresh = store.state.output else { throw DecideScenarioError("No fresh result.") }
        try require(
            !fresh.reusedDocument && store.state.confirmations.isEmpty,
            "A fresh result retained a confirmation or reuse flag.")
        let difference = try maximumDifference(reused, fresh)
        try require(difference <= 0.011, "Fresh and reused probabilities differ by \(difference).")
        let finalReview = fresh.rows.first { $0.reviewFirst } ?? fresh.rows[0]
        store.send(.confirm(finalReview.id, finalReview.candidate))
        store.send(.prepareExport)
        await harness.waitForExport()
        guard let exported = store.state.exportData else { throw DecideScenarioError("The final export is missing.") }
        try await files.write(exported, to: directory.appendingPathComponent("review.json"))
        try await files.write(Data(fresh.requestJSON.utf8), to: directory.appendingPathComponent("request.json"))
        try await files.write(Data(fresh.responseJSON.utf8), to: directory.appendingPathComponent("response.json"))
        let report: [String: Any] = [
            "passed": true,
            "items": fresh.rows.count,
            "inputTokens": fresh.inputTokens,
            "reusedTokens": reused.reusedTokens,
            "initialSeconds": initial.contextSeconds + initial.seconds,
            "reuseSeconds": reused.seconds,
            "freshSeconds": fresh.seconds,
            "maximumFreshReuseProbabilityDifference": difference,
            "modelVersion": fresh.modelVersion,
            "reviewerConfirmations": store.state.confirmations.count,
            "sourceSearchMatched": true,
        ]
        try await files.write(
            JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]),
            to: directory.appendingPathComponent("report.json"))
        store.send(.ended)
    }

    private static func maximumDifference(_ first: DecideOutput, _ second: DecideOutput) throws -> Float {
        try require(first.rows.count == second.rows.count, "Result sizes differ.")
        var maximum: Float = 0
        for (lhs, rhs) in zip(first.rows, second.rows) {
            try require(lhs.key.utf8.elementsEqual(rhs.key.utf8), "Question order differs.")
            try require(lhs.probabilities.count == rhs.probabilities.count, "Option counts differ.")
            for (a, b) in zip(lhs.probabilities, rhs.probabilities) {
                try require(a.option.utf8.elementsEqual(b.option.utf8), "Option order differs.")
                maximum = max(maximum, abs(a.probability - b.probability))
            }
        }
        return maximum
    }

    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw DecideScenarioError(message) }
    }
}

private struct DecideScenarioError: Error, LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

@MainActor
private final class DecideScenarioHarness {
    let store: Store<DecideFeature>
    private let completion: DecideScenarioCompletion

    init(service: any DecideService) {
        let runner = DecideRunner(service: service)
        let completion = DecideScenarioCompletion()
        self.completion = completion
        store = Store(initialState: DecideState()) { effect, send in
            await runner.run(effect, send: send)
            completion.signal()
        }
    }

    func settle() async {
        while store.state.busy || store.state.loadingExamples || counting {
            await completion.wait()
        }
    }

    func waitForExport() async {
        while store.state.exportData == nil && store.state.message == nil { await completion.wait() }
    }

    private var counting: Bool {
        store.state.model != nil && store.state.input.problem == nil
            && store.state.inputTokens == nil && store.state.countError == nil && store.state.message == nil
    }
}

@MainActor
private final class DecideScenarioCompletion {
    private var pending = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func signal() {
        if waiters.isEmpty { pending += 1 } else { waiters.removeFirst().resume() }
    }

    func wait() async {
        if pending > 0 { pending -= 1 } else { await withCheckedContinuation { waiters.append($0) } }
    }
}
