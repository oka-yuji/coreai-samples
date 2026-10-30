import ConversationRuntime
import Foundation
import Testing

struct SecretStoreTests {
    private let key = "agent.braveAPIKey"

    @Test("Setting the key writes it to the store and not to UserDefaults", arguments: [nil, "old-key"] as [String?])
    func saveWritesStore(legacy: String?) throws {
        try withDefaults { defaults in
            defaults.set(legacy, forKey: key)
            let store = FakeSecretStore()
            store.save("new-key", replacing: key, in: defaults, persisting: true)
            #expect(store.value == "new-key")
            #expect(store.calls == ["write"])
            #expect(defaults.object(forKey: key) == nil)
        }
    }

    @Test(
        "A legacy key in UserDefaults moves to the store at launch and is removed from UserDefaults",
        arguments: [nil, ""] as [String?])
    func legacyKeyMoves(stored: String?) throws {
        try withDefaults { defaults in
            defaults.set("legacy-key", forKey: key)
            let store = FakeSecretStore(stored)
            #expect(store.load(migrating: key, from: defaults, persisting: true) == "legacy-key")
            #expect(store.value == "legacy-key")
            #expect(store.calls == ["read", "write"])
            #expect(defaults.object(forKey: key) == nil)
        }
    }

    @Test("A key already in the store is used and the legacy key is removed from UserDefaults")
    func storedKeyWins() throws {
        try withDefaults { defaults in
            defaults.set("legacy-key", forKey: key)
            let store = FakeSecretStore("stored-key")
            #expect(store.load(migrating: key, from: defaults, persisting: true) == "stored-key")
            #expect(store.value == "stored-key")
            #expect(store.calls == ["read"])
            #expect(defaults.object(forKey: key) == nil)
        }
    }

    @Test("An empty legacy value is removed from UserDefaults without writing the store")
    func emptyLegacyRemoved() throws {
        try withDefaults { defaults in
            defaults.set("", forKey: key)
            let store = FakeSecretStore()
            #expect(store.load(migrating: key, from: defaults, persisting: true).isEmpty)
            #expect(store.calls == ["read"])
            #expect(defaults.object(forKey: key) == nil)
        }
    }

    @Test(
        "When the store cannot be read or written at launch, the legacy key stays in UserDefaults",
        arguments: ["read", "write"])
    func failedMigrationKeepsLegacy(failing: String) throws {
        try withDefaults { defaults in
            defaults.set("legacy-key", forKey: key)
            let store = FakeSecretStore(failing: [failing])
            #expect(store.load(migrating: key, from: defaults, persisting: true) == "legacy-key")
            #expect(store.value == nil)
            #expect(defaults.string(forKey: key) == "legacy-key")
        }
    }

    @Test(
        "When saving to the store fails, the legacy key stays in UserDefaults",
        arguments: zip(["new-key", ""], ["write", "delete"]))
    func failedSaveKeepsLegacy(value: String, failing: String) throws {
        try withDefaults { defaults in
            defaults.set("legacy-key", forKey: key)
            let store = FakeSecretStore("stored-key", failing: [failing])
            store.save(value, replacing: key, in: defaults, persisting: true)
            #expect(store.value == "stored-key")
            #expect(defaults.string(forKey: key) == "legacy-key")
        }
    }

    @Test("Clearing the key deletes the stored item")
    func clearingDeletes() throws {
        try withDefaults { defaults in
            let store = FakeSecretStore("stored-key")
            store.save("", replacing: key, in: defaults, persisting: true)
            #expect(store.value == nil)
            #expect(store.calls == ["delete"])
            #expect(defaults.object(forKey: key) == nil)
        }
    }

    @Test("Headless settings neither read nor write the store and leave UserDefaults unchanged")
    func headlessLeavesStore() throws {
        try withDefaults { defaults in
            defaults.set("legacy-key", forKey: key)
            let store = FakeSecretStore("stored-key")
            #expect(store.load(migrating: key, from: defaults, persisting: false).isEmpty)
            store.save("typed-key", replacing: key, in: defaults, persisting: false)
            store.save("", replacing: key, in: defaults, persisting: false)
            #expect(store.calls.isEmpty)
            #expect(store.value == "stored-key")
            #expect(defaults.string(forKey: key) == "legacy-key")
        }
    }

    private func withDefaults(_ body: (UserDefaults) throws -> Void) throws {
        let name = "secret-store-tests-" + UUID().uuidString
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        try body(defaults)
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
    struct Failure: Error {}

    var value: String?
    private let failing: Set<String>
    private(set) var calls: [String] = []

    init(_ value: String? = nil, failing: Set<String> = []) {
        self.value = value
        self.failing = failing
    }

    func read() throws -> String? {
        try record("read")
        return value
    }

    func write(_ value: String) throws {
        try record("write")
        self.value = value
    }

    func delete() throws {
        try record("delete")
        value = nil
    }

    private func record(_ call: String) throws {
        calls.append(call)
        if failing.contains(call) { throw Failure() }
    }
}
