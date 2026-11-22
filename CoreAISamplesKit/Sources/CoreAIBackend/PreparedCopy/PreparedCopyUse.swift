import Darwin
import Foundation

final class PreparedCopyUse: Sendable {
    private let descriptor: Int32

    private init(descriptor: Int32) {
        self.descriptor = descriptor
    }

    deinit {
        close(descriptor)
    }

    static func take(_ file: URL, exclusive: Bool) throws -> PreparedCopyUse? {
        let descriptor = try PreparedCopyDescriptor.make(file)
        guard flock(descriptor, (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB) == 0 else {
            close(descriptor)
            return nil
        }
        return PreparedCopyUse(descriptor: descriptor)
    }

    func share() throws {
        try PreparedCopyDescriptor.lock(descriptor, LOCK_SH)
    }
}
