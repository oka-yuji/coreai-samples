import Foundation
import Security

public protocol SecretStore {
    func read() throws -> String?
    func write(_ value: String) throws
    func delete() throws
}

extension SecretStore {
    public func load(migrating key: String, from defaults: UserDefaults, persisting: Bool) -> String {
        guard persisting else { return "" }
        let legacy = defaults.string(forKey: key) ?? ""
        let stored: String
        do {
            stored = try read() ?? ""
            if stored.isEmpty, !legacy.isEmpty { try write(legacy) }
        } catch {
            return legacy
        }
        defaults.removeObject(forKey: key)
        return stored.isEmpty ? legacy : stored
    }

    public func save(_ value: String, replacing key: String, in defaults: UserDefaults, persisting: Bool) {
        guard persisting else { return }
        do {
            if value.isEmpty { try delete() } else { try write(value) }
        } catch {
            return
        }
        defaults.removeObject(forKey: key)
    }
}

public struct KeychainSecretStore: SecretStore, Sendable {
    public struct Failure: Error, Equatable {
        public var status: OSStatus
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
