import ConversationFeature
import Dispatch
import Foundation
import Security

public protocol SecretStore: Sendable {
    func read() throws -> String?
    func write(_ value: String) throws
    func delete() throws
}

public actor SearchKeyVault {
    private let store: any SecretStore
    private let queue = DispatchSerialQueue(label: "com.coreai-samples.search-key")

    public nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

    public init(store: any SecretStore) {
        self.store = store
    }

    public func load(legacy: String) -> SearchKeyFeature.Load {
        let stored: String
        do {
            stored = try store.read() ?? ""
        } catch {
            return .unreadable(Self.failure(.read, error))
        }
        if !stored.isEmpty || legacy.isEmpty { return .stored(stored) }
        do {
            try store.write(legacy)
        } catch {
            return .unmoved(legacy, Self.failure(.move, error))
        }
        return .stored(legacy)
    }

    public func save(_ value: String) -> Result<String, SearchKeyFeature.Failure> {
        do {
            if value.isEmpty { try store.delete() } else { try store.write(value) }
        } catch {
            return .failure(Self.failure(value.isEmpty ? .delete : .write, error))
        }
        return .success(value)
    }

    public func flush() {}

    private static func failure(_ operation: SearchKeyFeature.Failure.Operation, _ error: any Error)
        -> SearchKeyFeature.Failure
    {
        SearchKeyFeature.Failure(operation, reason: error.localizedDescription)
    }
}

@MainActor
public struct SearchKeyRunner {
    public let vault: SearchKeyVault?
    private let defaults: UserDefaults
    private let legacyKey: String

    public init(vault: SearchKeyVault?, defaults: UserDefaults, legacyKey: String) {
        self.vault = vault
        self.defaults = defaults
        self.legacyKey = legacyKey
    }

    public func run(_ effect: SearchKeyFeature.Effect, _ send: (SearchKeyFeature.Action) -> Void) async {
        switch effect {
        case .load(let ticket):
            guard let vault else {
                send(.loaded(ticket: ticket, .stored("")))
                return
            }
            let load = await vault.load(legacy: defaults.string(forKey: legacyKey) ?? "")
            if case .stored = load { defaults.removeObject(forKey: legacyKey) }
            send(.loaded(ticket: ticket, load))
        case .save(let value, let ticket):
            guard let vault else {
                send(.saved(ticket: ticket, .success(value)))
                return
            }
            let result = await vault.save(value)
            if case .success = result { defaults.removeObject(forKey: legacyKey) }
            send(.saved(ticket: ticket, result))
        }
    }
}

public struct KeychainSecretStore: SecretStore {
    public struct Failure: Error, Equatable, LocalizedError {
        public var status: OSStatus

        public init(status: OSStatus) {
            self.status = status
        }

        public var errorDescription: String? {
            let message = SecCopyErrorMessageString(status, nil).map { $0 as String } ?? "Keychain error"
            return "\(message) (\(status))"
        }
    }

    public let service: String
    public let account: String

    public init(service: String, account: String) {
        self.service = service
        self.account = account
    }

    public func read() throws -> String? {
        var search = query
        search[kSecReturnData as String] = true
        search[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(search as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess else { throw Failure(status: status) }
        guard let data = result as? Data, let value = String(data: data, encoding: .utf8) else {
            throw Failure(status: errSecDecode)
        }
        return value
    }

    public func write(_ value: String) throws {
        let data = Data(value.utf8)
        var status = SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecItemNotFound {
            var item = query
            item[kSecValueData as String] = data
            status = SecItemAdd(item as CFDictionary, nil)
        }
        guard status == errSecSuccess else { throw Failure(status: status) }
    }

    public func delete() throws {
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw Failure(status: status) }
    }

    private var query: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }
}
