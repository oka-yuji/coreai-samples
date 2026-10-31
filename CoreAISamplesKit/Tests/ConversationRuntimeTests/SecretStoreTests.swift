import ConversationFeature
import ConversationRuntime
import FeatureStore
import Foundation
import Observation
import Security
import Synchronization
import Testing

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct SecretStoreTests {
    private let key = "agent.braveAPIKey"

    @Test("Setting the key writes it to the store and not to UserDefaults", arguments: [nil, "old-key"] as [String?])
    func saveWritesStore(legacy: String?) async throws {
        try await withDefaults { defaults in
            defaults.set(legacy, forKey: key)
            let store = FakeSecretStore()
            let keys = makeStore(store, defaults)
            await load(keys)
            keys.send(.draftChanged("new-key"))
            keys.send(.saveRequested)
            await settle(keys)
            #expect(store.value == "new-key" && keys.state.saved == "new-key" && keys.state.status == .saved)
            #expect(store.calls.last == "write")
            #expect(defaults.object(forKey: key) == nil)
        }
    }

    @Test(
        "A legacy key in UserDefaults moves to the store at launch and is removed from UserDefaults",
        arguments: [nil, ""] as [String?])
    func legacyKeyMoves(stored: String?) async throws {
        try await withDefaults { defaults in
            defaults.set("legacy-key", forKey: key)
            let store = FakeSecretStore(stored)
            let keys = makeStore(store, defaults)
            await load(keys)
            #expect(keys.state.saved == "legacy-key" && keys.state.status == .saved)
            #expect(store.value == "legacy-key" && store.calls == ["read", "write"])
            #expect(defaults.object(forKey: key) == nil)
        }
    }

    @Test("A key already in the store is used and the legacy key is removed from UserDefaults")
    func storedKeyWins() async throws {
        try await withDefaults { defaults in
            defaults.set("legacy-key", forKey: key)
            let store = FakeSecretStore("stored-key")
            let keys = makeStore(store, defaults)
            await load(keys)
            #expect(keys.state.saved == "stored-key" && store.value == "stored-key" && store.calls == ["read"])
            #expect(defaults.object(forKey: key) == nil)
        }
    }

    @Test("An empty legacy value is removed from UserDefaults without writing the store")
    func emptyLegacyRemoved() async throws {
        try await withDefaults { defaults in
            defaults.set("", forKey: key)
            let store = FakeSecretStore()
            let keys = makeStore(store, defaults)
            await load(keys)
            #expect(keys.state.status == .empty && store.calls == ["read"])
            #expect(defaults.object(forKey: key) == nil)
        }
    }

    @Test(
        "When the store cannot be read or written at launch, the legacy key stays and the failure is shown",
        arguments: [("read", SearchKeyFeature.Failure.Operation.read), ("write", .move)])
    func failedMigrationKeepsLegacy(failing: String, operation: SearchKeyFeature.Failure.Operation) async throws {
        try await withDefaults { defaults in
            defaults.set("legacy-key", forKey: key)
            let store = FakeSecretStore(failing: [failing])
            let keys = makeStore(store, defaults)
            await load(keys)
            #expect(keys.state.status == .failed(SearchKeyFeature.Failure(operation, reason: "Injected failure")))
            #expect(keys.state.saved == (operation == .read ? "" : "legacy-key"))
            #expect(store.value == nil)
            #expect(defaults.string(forKey: key) == "legacy-key")
        }
    }

    @Test("When retrying the move keeps failing, the legacy key stays and the move can be retried again")
    func repeatedMoveFailuresKeepLegacy() async throws {
        try await withDefaults { defaults in
            defaults.set("legacy-key", forKey: key)
            let store = FakeSecretStore(failing: ["write"])
            let keys = makeStore(store, defaults)
            await load(keys)
            for _ in 0..<2 {
                keys.send(.saveRequested)
                await settle(keys)
            }
            #expect(keys.state.status == .failed(SearchKeyFeature.Failure(.move, reason: "Injected failure")))
            #expect(keys.state.saved == "legacy-key" && keys.state.retry == .saveRequested && !keys.state.canLeave)
            #expect(store.calls == ["read", "write", "write", "write"] && store.value == nil)
            #expect(defaults.string(forKey: key) == "legacy-key")
        }
    }

    @Test(
        "When saving fails, the saved key, the draft and the legacy key are kept and the reason is returned",
        arguments: zip(["new-key", ""], [SearchKeyFeature.Failure.Operation.write, .delete]))
    func failedSaveKeepsState(value: String, operation: SearchKeyFeature.Failure.Operation) async throws {
        try await withDefaults { defaults in
            let store = FakeSecretStore("stored-key", failing: [operation == .write ? "write" : "delete"])
            let keys = makeStore(store, defaults)
            await load(keys)
            defaults.set("legacy-key", forKey: key)
            keys.send(.draftChanged(value))
            keys.send(.saveRequested)
            await settle(keys)
            #expect(keys.state.status == .failed(SearchKeyFeature.Failure(operation, reason: "Injected failure")))
            #expect(keys.state.saved == "stored-key" && keys.state.draft == value && keys.state.canSave)
            #expect(store.value == "stored-key")
            #expect(defaults.string(forKey: key) == "legacy-key")
        }
    }

    @Test("Clearing the key deletes the stored item")
    func clearingDeletes() async throws {
        try await withDefaults { defaults in
            let store = FakeSecretStore("stored-key")
            let keys = makeStore(store, defaults)
            await load(keys)
            keys.send(.draftChanged(""))
            keys.send(.saveRequested)
            await settle(keys)
            #expect(store.value == nil && store.calls == ["read", "delete"])
            #expect(keys.state.saved.isEmpty && keys.state.status == .empty)
        }
    }

    @Test("Headless settings keep the key in memory and neither read nor write the store or UserDefaults")
    func headlessLeavesStore() async throws {
        try await withDefaults { defaults in
            defaults.set("legacy-key", forKey: key)
            let keys = makeStore(nil, defaults)
            await load(keys)
            #expect(keys.state.saved.isEmpty)
            keys.send(.draftChanged("typed-key"))
            keys.send(.saveRequested)
            await settle(keys)
            #expect(keys.state.saved == "typed-key")
            #expect(defaults.string(forKey: key) == "legacy-key")
        }
    }

    @Test("Keychain calls run one at a time off the main actor, and the main actor keeps running during a save")
    func callsRunOffMainActor() async throws {
        try await withDefaults { defaults in
            let store = FakeSecretStore(gated: true)
            defer { store.release() }
            await withTaskCancellationHandler {
                let vault = SearchKeyVault(store: store)
                let runner = SearchKeyRunner(vault: vault, defaults: defaults, legacyKey: key)
                let keys = Store<SearchKeyFeature>(initialState: SearchKeyFeature.State(), run: runner.run)
                await load(keys)
                keys.send(.draftChanged("new-key"))
                keys.send(.saveRequested)
                for await call in store.started where call == "write" { break }
                #expect(keys.state.phase == .saving)
                async let reread = vault.load(legacy: "")
                store.release()
                #expect(await reread == .stored("new-key"))
                await settle(keys)
                #expect(keys.state.saved == "new-key" && store.calls == ["read", "write", "read"])
                #expect(!store.ranOnMainThread && store.peak == 1)
            } onCancel: {
                store.release()
            }
        }
    }

    @Test("A Keychain failure explains its status")
    func keychainFailureDescription() {
        let description = KeychainSecretStore.Failure(status: errSecItemNotFound).localizedDescription
        #expect(description.hasSuffix(" (\(errSecItemNotFound))") && description.count > 10)
    }

    private func makeStore(_ store: FakeSecretStore?, _ defaults: UserDefaults) -> Store<SearchKeyFeature> {
        let runner = SearchKeyRunner(vault: store.map { SearchKeyVault(store: $0) }, defaults: defaults, legacyKey: key)
        return Store(initialState: SearchKeyFeature.State(), run: runner.run)
    }

    private func load(_ keys: Store<SearchKeyFeature>) async {
        keys.send(.loadRequested)
        await settle(keys)
    }

    private func settle(_ keys: Store<SearchKeyFeature>) async {
        for await phase in Observations({ keys.state.phase }) where phase == .ready { return }
    }

    private func withDefaults(_ body: (UserDefaults) async throws -> Void) async throws {
        let name = "secret-store-tests-" + UUID().uuidString
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        try await body(defaults)
    }
}

struct KeychainSecretStoreTests {
    static let service = "com.coreai-samples.keychain-test"

    @Test(
        "The Keychain store saves, reads, overwrites and deletes a test item and leaves nothing behind",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_KEYCHAIN_TEST"] == "1"))
    func roundTrip() throws {
        let store = KeychainSecretStore(service: Self.service, account: UUID().uuidString)
        defer { try? store.delete() }
        try #require(try store.read() == nil)
        try store.write("first-key")
        #expect(try store.read() == "first-key")
        try store.write("second-key")
        #expect(try store.read() == "second-key")
        try store.delete()
        #expect(try store.read() == nil)
        try store.delete()
    }
}

private final class FakeSecretStore: SecretStore {
    struct Failure: LocalizedError {
        var errorDescription: String? { "Injected failure" }
    }

    private struct Record {
        var value: String?
        var calls: [String] = []
        var onMainThread = false
        var active = 0
        var peak = 0
    }

    let started: AsyncStream<String>
    private let start: AsyncStream<String>.Continuation
    private let record: Mutex<Record>
    private let failing: Set<String>
    private let gate: DispatchSemaphore?

    init(_ value: String? = nil, failing: Set<String> = [], gated: Bool = false) {
        (started, start) = AsyncStream.makeStream()
        record = Mutex(Record(value: value))
        self.failing = failing
        gate = gated ? DispatchSemaphore(value: 0) : nil
    }

    var value: String? { record.withLock { $0.value } }
    var calls: [String] { record.withLock { $0.calls } }
    var ranOnMainThread: Bool { record.withLock { $0.onMainThread } }
    var peak: Int { record.withLock { $0.peak } }

    func release() { gate?.signal() }

    func read() throws -> String? {
        try enter("read")
        defer { leave() }
        return value
    }

    func write(_ value: String) throws {
        try enter("write")
        defer { leave() }
        record.withLock { $0.value = value }
    }

    func delete() throws {
        try enter("delete")
        defer { leave() }
        record.withLock { $0.value = nil }
    }

    private func enter(_ call: String) throws {
        let onMainThread = Thread.isMainThread
        record.withLock {
            $0.calls.append(call)
            $0.onMainThread = $0.onMainThread || onMainThread
            $0.active += 1
            $0.peak = max($0.peak, $0.active)
        }
        start.yield(call)
        if call == "write" { gate?.wait() }
        guard failing.contains(call) else { return }
        leave()
        throw Failure()
    }

    private func leave() { record.withLock { $0.active -= 1 } }
}
