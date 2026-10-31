import AppKit

@MainActor
final class ConversationLifetime: NSObject, NSApplicationDelegate {
    private var hosts: [UUID: ConversationHost] = [:]
    private var closing: Set<UUID> = []
    private var waiting: Set<UUID> = []

    func register(_ id: UUID, host: ConversationHost) {
        hosts[id] = host
    }

    func close(_ id: UUID) {
        guard let host = hosts[id] else { return }
        closing.insert(id)
        host.store.send(.terminationRequested)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard waiting.isEmpty else { return .terminateLater }
        guard !hosts.isEmpty else { return .terminateNow }
        waiting = Set(hosts.keys)
        for host in hosts.values {
            host.store.send(.terminationCancelled)
            host.store.send(.terminationRequested)
        }
        return .terminateLater
    }

    func finished(_ id: UUID, success: Bool) {
        if !waiting.isEmpty {
            guard waiting.remove(id) != nil else { return }
            if !success {
                waiting.removeAll()
                for host in hosts.values { host.store.send(.terminationCancelled) }
                NSApplication.shared.reply(toApplicationShouldTerminate: false)
                showFailure(id)
                return
            } else if waiting.isEmpty {
                NSApplication.shared.reply(toApplicationShouldTerminate: true)
            }
        }
        if closing.contains(id) {
            if success {
                closing.remove(id)
                hosts[id] = nil
            } else {
                showFailure(id)
            }
        }
    }

    private func showFailure(_ id: UUID) {
        guard let host = hosts[id] else { return }
        guard host.preferences.searchKey.state.canLeave else {
            showKeyFailure(id, host)
            return
        }
        let alert = NSAlert()
        alert.messageText = String(localized: "Could not save the conversation")
        alert.informativeText = String(
            localized:
                "The app is still open. You can save a separate copy, or fix the save error and try quitting again.")
        if case .failed(let message) = host.state.persistence {
            alert.informativeText += "\n\n" + host.state.title + "\n" + message
        }
        alert.addButton(withTitle: String(localized: "Keep app open"))
        alert.addButton(withTitle: String(localized: "Save a copy"))
        if alert.runModal() == .alertSecondButtonReturn {
            host.store.send(.terminationCancelled)
            host.saveCopy()
            if closing.contains(id) { host.store.send(.terminationRequested) }
        }
    }

    private func showKeyFailure(_ id: UUID, _ host: ConversationHost) {
        let keys = host.preferences.searchKey
        let alert = NSAlert()
        alert.messageText = String(localized: "Could not save the Brave Search API key")
        alert.informativeText = String(
            localized: "The app is still open. You can retry, or fix the keychain error and try quitting again.")
        if let failure = keys.state.failure { alert.informativeText += "\n\n" + failure.reason }
        alert.addButton(withTitle: String(localized: "Keep app open"))
        alert.addButton(withTitle: String(localized: "Retry"))
        guard alert.runModal() == .alertSecondButtonReturn, let retry = keys.state.retry else { return }
        host.store.send(.terminationCancelled)
        keys.send(retry)
        if closing.contains(id) { host.store.send(.terminationRequested) }
    }
}
