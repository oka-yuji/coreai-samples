#if os(macOS)
    import Darwin
    import Foundation
    import Synchronization

    final class MacSpeechRecordingCleanup: Sendable {
        static let shared = MacSpeechRecordingCleanup(remove: { url in try unlinkFile(url) })

        private let remove: @Sendable (URL) throws -> Void
        private let pending = Mutex<[URL]>([])

        init(remove: @escaping @Sendable (URL) throws -> Void) {
            self.remove = remove
        }

        func discard(_ ownedURL: URL) {
            pending.withLock { urls in
                if attempt(ownedURL) {
                    urls.removeAll { $0 == ownedURL }
                } else if !urls.contains(ownedURL) {
                    urls.append(ownedURL)
                }
            }
        }

        func retryPending() -> Bool {
            pending.withLock { urls in
                urls.removeAll { attempt($0) }
                return urls.isEmpty
            }
        }

        private func attempt(_ url: URL) -> Bool {
            do {
                try remove(url)
                return true
            } catch {
                return Self.isMissing(error)
            }
        }

        private static func isMissing(_ error: any Error) -> Bool {
            if let posix = error as? POSIXError { return posix.code == .ENOENT }
            if let cocoa = error as? CocoaError { return cocoa.code == .fileNoSuchFile }
            return false
        }

        private static func unlinkFile(_ url: URL) throws {
            guard url.isFileURL else { throw POSIXError(.EINVAL) }
            let code = url.withUnsafeFileSystemRepresentation { path -> Int32 in
                guard let path else { return EINVAL }
                return unlink(path) == 0 ? 0 : errno
            }
            guard code != 0 else { return }
            throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
        }
    }
#endif
