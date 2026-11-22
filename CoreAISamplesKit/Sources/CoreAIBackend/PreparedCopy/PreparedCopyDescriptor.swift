import Darwin
import Foundation

enum PreparedCopyDescriptor {
    static func make(_ file: URL) throws -> Int32 {
        let descriptor = open(file.path(percentEncoded: false), O_CREAT | O_RDWR | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else { throw failure() }
        return descriptor
    }

    static func lock(_ descriptor: Int32, _ operation: Int32) throws {
        while flock(descriptor, operation) != 0 {
            guard errno == EINTR else { throw failure() }
        }
    }

    static func failure() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}
