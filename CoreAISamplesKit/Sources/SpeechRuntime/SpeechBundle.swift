#if os(macOS) || os(iOS)
    import CryptoKit
    import Foundation
    import SpeechCore
    import SpeechAssets

    struct SpeechBundleLayout {
        let files: [IrodoriBundleInventory.File]
        let assets: [String]

        static func irodori(manifest: Data) throws -> Self {
            let profile = try IrodoriBundleInventory.profile(in: manifest)
            guard hexDigest(SHA256.hash(data: manifest)) == IrodoriBundleInventory.manifestSHA256(for: profile) else {
                throw SpeechError.invalidAsset("The Irodori manifest does not match the selected model profile.")
            }
            return Self(
                files: try IrodoriBundleInventory.files(profile: profile),
                assets: try IrodoriBundleInventory.assetSpecifications(profile: profile).map {
                    "artifacts/" + $0.stem + ".aimodel"
                })
        }
    }

    struct ValidatedSpeechBundle: Sendable {
        struct CloneFailure: Error {
            let code: Int32
        }

        static var inPlaceWhenCloneFails: Bool {
            #if os(iOS)
                false
            #else
                true
            #endif
        }

        static var defaultRoot: URL { defaultRoot(for: .irodori) }

        static func defaultRoot(for names: SpeechBundleNames) -> URL {
            URL.cachesDirectory.appending(
                path: (Bundle.main.bundleIdentifier ?? ProcessInfo.processInfo.processName) + "/"
                    + names.verifiedFolder,
                directoryHint: .isDirectory)
        }

        let directory: URL
        let assets: [URL]
        let tokenizer: URL

        static func open(
            _ source: URL, copyingInto copies: URL? = nil,
            layout: (Data) throws -> SpeechBundleLayout = SpeechBundleLayout.irodori(manifest:),
            names: SpeechBundleNames = .irodori,
            inPlaceWhenCloneFails: Bool = Self.inPlaceWhenCloneFails,
            clone: (Int32, String) -> Int32 = Self.cloneFile(_:to:)
        ) throws -> Self {
            let root = copies ?? defaultRoot(for: names)
            let incoming = root.appending(path: "incoming-" + UUID().uuidString, directoryHint: .isDirectory)
            do {
                try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                let lock = Darwin.open(
                    root.appending(path: ".lock").path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
                guard lock >= 0 else { throw CloneFailure(code: errno) }
                defer { close(lock) }
                guard flock(lock, LOCK_EX) == 0 else {
                    throw CloneFailure(code: errno)
                }
                removeLeftovers(root)
                guard mkdir(incoming.path, 0o700) == 0 else {
                    throw CloneFailure(code: errno)
                }
                defer { try? FileManager.default.removeItem(at: incoming) }
                let (spec, key) = try verify(source, into: incoming, layout: layout, names: names, clone: clone)
                let target = root.appending(path: key, directoryHint: .isDirectory)
                let installed =
                    rename(incoming.path, target.path) == 0 || isComplete(target, spec.files, names)
                    || renamex_np(incoming.path, target.path, UInt32(RENAME_SWAP)) == 0
                guard installed else { throw CloneFailure(code: errno) }
                return bundle(at: target, spec, names)
            } catch is CloneFailure where inPlaceWhenCloneFails {
                let spec = try verify(source, into: nil, layout: layout, names: names, clone: clone).spec
                return bundle(at: source, spec, names)
            } catch let failure as CloneFailure {
                throw SpeechError.invalidAsset(
                    "The model folder could not be copied for verification (error \(failure.code)).")
            }
        }

        static func cloneFile(_ descriptor: Int32, to path: String) -> Int32 {
            fclonefileat(descriptor, AT_FDCWD, path, 0) == 0 ? 0 : errno
        }

        private static func verify(
            _ source: URL, into incoming: URL?, layout: (Data) throws -> SpeechBundleLayout, names: SpeechBundleNames,
            clone: (Int32, String) -> Int32
        ) throws -> (spec: SpeechBundleLayout, key: String) {
            try Task.checkCancellation()
            let input = try folder(source, names)
            defer { close(input) }
            if let incoming { try cloneFiles([names.manifest], from: input, into: incoming, names, clone) }
            let root = try folder(incoming ?? source, names)
            defer { close(root) }
            let manifest = try regularFile(names.manifest, in: root, names).handle.read(upToCount: 65_536) ?? Data()
            let spec = try layout(manifest)
            if let incoming {
                try cloneFiles(
                    spec.files.map(\.path).filter { $0 != names.manifest }, from: input, into: incoming, names, clone)
            }
            for file in spec.files {
                let (handle, status) = try regularFile(file.path, in: root, names)
                guard Int64(status.st_size) == file.bytes, try hashFile(handle) == file.sha256 else {
                    throw mismatch(file.path, names)
                }
            }
            return (spec, verifiedKey(manifest: manifest, files: spec.files, names: names))
        }

        static func verifiedKey(
            manifest: Data, files: [IrodoriBundleInventory.File], names: SpeechBundleNames
        ) -> String {
            guard let scheme = names.verifiedKeyScheme else { return hexDigest(SHA256.hash(data: manifest)) }
            var hash = SHA256()
            let ordered = files.sorted { $0.path.utf8.lexicographicallyPrecedes($1.path.utf8) }
            for field in [scheme] + ordered.flatMap({ [$0.path, String($0.bytes), $0.sha256] }) {
                withUnsafeBytes(of: UInt64(field.utf8.count).bigEndian) { hash.update(bufferPointer: $0) }
                hash.update(data: Data(field.utf8))
            }
            return hexDigest(hash.finalize())
        }

        private static func isComplete(
            _ copy: URL, _ files: [IrodoriBundleInventory.File], _ names: SpeechBundleNames
        ) -> Bool {
            guard let root = try? folder(copy, names) else { return false }
            defer { close(root) }
            return files.allSatisfy { file in
                (try? regularFile(file.path, in: root, names)).map { Int64($0.status.st_size) == file.bytes } ?? false
            }
        }

        private static func bundle(at base: URL, _ spec: SpeechBundleLayout, _ names: SpeechBundleNames) -> Self {
            Self(
                directory: base, assets: spec.assets.map { base.appending(path: $0, directoryHint: .isDirectory) },
                tokenizer: base.appending(path: names.tokenizer))
        }

        private static func removeLeftovers(_ root: URL) {
            guard let children = try? FileManager.default.contentsOfDirectory(atPath: root.path) else { return }
            for name in children {
                guard name.hasPrefix("incoming-") else { continue }
                var status = stat()
                guard lstat(root.appending(path: name).path, &status) == 0 else { continue }
                guard status.st_mode & S_IFMT == S_IFDIR else { continue }
                let locked = UInt32(UF_IMMUTABLE | UF_APPEND)
                let folder = root.appending(path: name)
                for relative in (try? FileManager.default.subpathsOfDirectory(atPath: folder.path)) ?? [] {
                    let path = folder.appending(path: relative).path
                    var entry = stat()
                    guard lstat(path, &entry) == 0 else { continue }
                    if entry.st_flags & locked != 0 {
                        lchflags(path, entry.st_flags & ~locked)
                    }
                }
                try? FileManager.default.removeItem(at: folder)
            }
        }

        private static func cloneFiles(
            _ paths: [String], from input: Int32, into copy: URL, _ names: SpeechBundleNames,
            _ clone: (Int32, String) -> Int32
        ) throws {
            let locked = UInt32(UF_IMMUTABLE | UF_APPEND)
            for path in paths {
                let handle = try regularFile(path, in: input, names).handle
                var parent = copy
                for part in path.split(separator: "/").dropLast() {
                    parent.append(path: String(part), directoryHint: .isDirectory)
                    guard mkdir(parent.path, 0o700) == 0 || errno == EEXIST else { throw CloneFailure(code: errno) }
                }
                let target = copy.appending(path: path).path
                let code = clone(handle.fileDescriptor, target)
                guard code == 0 else { throw CloneFailure(code: code) }
                var copied = stat()
                guard lstat(target, &copied) == 0,
                    copied.st_flags & locked == 0 || chflags(target, copied.st_flags & ~locked) == 0
                else { throw CloneFailure(code: errno) }
            }
        }

        private static func folder(_ url: URL, _ names: SpeechBundleNames) throws -> Int32 {
            let descriptor = Darwin.open(url.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
            guard descriptor >= 0 else {
                throw SpeechError.invalidAsset("Select a supported \(names.model) model folder.")
            }
            return descriptor
        }

        private static func regularFile(
            _ path: String, in root: Int32, _ names: SpeechBundleNames
        ) throws -> (handle: FileHandle, status: stat) {
            let parts = path.split(separator: "/").map(String.init)
            guard let name = parts.last else { throw mismatch(path, names) }
            var directory = dup(root)
            guard directory >= 0 else { throw mismatch(path, names) }
            defer { close(directory) }
            for part in parts.dropLast() {
                let next = openat(directory, part, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                guard next >= 0 else { throw mismatch(path, names) }
                close(directory)
                directory = next
            }
            let descriptor = openat(directory, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
            guard descriptor >= 0 else { throw mismatch(path, names) }
            let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
            var status = stat()
            guard fstat(descriptor, &status) == 0, status.st_mode & S_IFMT == S_IFREG else {
                throw mismatch(path, names)
            }
            return (handle, status)
        }

        private static func mismatch(_ path: String, _ names: SpeechBundleNames) -> SpeechError {
            .invalidAsset("Model files do not match the verified \(names.model) version: " + path)
        }
    }

    func hashFile(_ handle: FileHandle) throws -> String {
        var hash = SHA256()
        while true {
            try Task.checkCancellation()
            let more = try autoreleasepool { () throws -> Bool in
                guard let bytes = try handle.read(upToCount: 4 * 1024 * 1024), !bytes.isEmpty else { return false }
                hash.update(data: bytes)
                return true
            }
            if !more { break }
        }
        return hexDigest(hash.finalize())
    }

    private func hexDigest(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
#endif
