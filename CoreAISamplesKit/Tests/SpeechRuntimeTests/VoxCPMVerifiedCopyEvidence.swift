#if os(macOS)
    import Darwin
    import Foundation
    import SpeechAssets

    @testable import SpeechRuntime

    struct VoxCPMVerifiedCopyEvidence {
        let key: String
        let assetPaths: [String]

        func matches(_ bundle: ValidatedSpeechBundle, source: URL, copies: URL) -> Bool {
            guard source.isFileURL, copies.isFileURL, bundle.directory.isFileURL,
                key.utf8.count == 64, key.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
                assetPaths.count == 7, Set(assetPaths).count == 7, bundle.assets.count == 7
            else { return false }
            let expected = copies.appending(path: key, directoryHint: .isDirectory).standardizedFileURL
            let real = copies.resolvingSymlinksInPath().standardizedFileURL.appending(
                path: key, directoryHint: .isDirectory)
            var original = stat()
            var copied = stat()
            var artifacts = stat()
            guard lstat(source.resolvingSymlinksInPath().path, &original) == 0, lstat(copies.path, &copied) == 0,
                copied.st_mode & S_IFMT == S_IFDIR,
                lstat(expected.path, &copied) == 0, copied.st_mode & S_IFMT == S_IFDIR,
                lstat(expected.appending(path: "artifacts").path, &artifacts) == 0,
                artifacts.st_mode & S_IFMT == S_IFDIR,
                original.st_dev != copied.st_dev || original.st_ino != copied.st_ino,
                Self.same(bundle.directory, expected, real, type: S_IFDIR)
            else { return false }
            for (url, path) in zip(bundle.assets, assetPaths) {
                guard Self.same(url, expected.appending(path: path), real.appending(path: path), type: S_IFDIR) else {
                    return false
                }
            }
            let tokenizer = VoxCPMBundleInventory.tokenizerPath
            return Self.same(
                bundle.tokenizer, expected.appending(path: tokenizer), real.appending(path: tokenizer), type: S_IFREG)
        }

        private static func same(_ url: URL, _ expected: URL, _ physical: URL, type: mode_t) -> Bool {
            var info = stat()
            return url.isFileURL && url.standardizedFileURL.path == expected.standardizedFileURL.path
                && url.resolvingSymlinksInPath().standardizedFileURL.path == physical.standardizedFileURL.path
                && lstat(url.path, &info) == 0 && info.st_mode & S_IFMT == type
        }
    }
#endif
