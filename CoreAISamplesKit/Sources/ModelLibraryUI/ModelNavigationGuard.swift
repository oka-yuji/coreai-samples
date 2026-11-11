#if os(iOS)
    import SwiftUI
    import UIKit

    struct ModelNavigationGuard: UIViewControllerRepresentable {
        func makeUIViewController(context: Context) -> Controller { Controller() }

        func updateUIViewController(_ controller: Controller, context: Context) { controller.synchronize() }

        static func dismantleUIViewController(_ controller: Controller, coordinator: ()) { controller.release() }

        final class Controller: UIViewController {
            private let owner = UUID()
            private var gestures: Set<ObjectIdentifier> = []
            private var ended = false

            override func viewDidAppear(_ animated: Bool) {
                super.viewDidAppear(animated)
                synchronize()
            }

            override func viewDidLayoutSubviews() {
                super.viewDidLayoutSubviews()
                synchronize()
            }

            func synchronize() {
                guard !ended else { return }
                let current = [
                    navigationController?.interactivePopGestureRecognizer,
                    navigationController?.interactiveContentPopGestureRecognizer,
                ].compactMap { $0 }
                let identifiers = Set(current.map(ObjectIdentifier.init))
                for identifier in gestures.subtracting(identifiers) {
                    NavigationGestures.release(identifier, owner: owner)
                }
                for gesture in current { NavigationGestures.acquire(gesture, owner: owner) }
                gestures = identifiers
            }

            func release() {
                ended = true
                for identifier in gestures { NavigationGestures.release(identifier, owner: owner) }
                gestures = []
            }
        }
    }

    @MainActor
    private enum NavigationGestures {
        private struct Entry {
            weak var gesture: UIGestureRecognizer?
            let enabled: Bool
            var owners: Set<UUID>
        }

        private static var entries: [ObjectIdentifier: Entry] = [:]

        static func acquire(_ gesture: UIGestureRecognizer, owner: UUID) {
            let identifier = ObjectIdentifier(gesture)
            if entries[identifier]?.gesture !== gesture {
                entries[identifier] = Entry(gesture: gesture, enabled: gesture.isEnabled, owners: [])
            }
            entries[identifier]?.owners.insert(owner)
            gesture.isEnabled = false
        }

        static func release(_ identifier: ObjectIdentifier, owner: UUID) {
            guard var entry = entries[identifier] else { return }
            entry.owners.remove(owner)
            if entry.owners.isEmpty {
                if let gesture = entry.gesture, !gesture.isEnabled { gesture.isEnabled = entry.enabled }
                entries.removeValue(forKey: identifier)
            } else {
                entries[identifier] = entry
            }
        }
    }
#endif
