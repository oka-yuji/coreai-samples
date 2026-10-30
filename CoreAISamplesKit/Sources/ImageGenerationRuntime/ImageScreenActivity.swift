import Foundation

#if os(iOS)
    import UIKit
#endif

@MainActor
final class ImageScreenActivity {
    #if os(iOS)
        static let shared = ImageScreenActivity(
            read: { UIApplication.shared.isIdleTimerDisabled },
            write: { UIApplication.shared.isIdleTimerDisabled = $0 })
    #else
        static let shared = ImageScreenActivity(read: { false }, write: { _ in })
    #endif

    private let read: () -> Bool
    private let write: (Bool) -> Void
    private var owners: Set<UUID> = []
    private var original: Bool?

    init(read: @escaping () -> Bool, write: @escaping (Bool) -> Void) {
        self.read = read
        self.write = write
    }

    func begin() -> UUID {
        let first = owners.isEmpty
        let id = UUID()
        owners.insert(id)
        if first {
            original = read()
            write(true)
        }
        return id
    }

    func end(_ id: UUID) {
        guard owners.remove(id) != nil, owners.isEmpty, let original else { return }
        self.original = nil
        write(original)
    }
}
