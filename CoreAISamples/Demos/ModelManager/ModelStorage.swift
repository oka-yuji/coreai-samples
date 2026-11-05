import Foundation
import ModelLibraryFeature
import ModelLibraryStorage

enum ModelStorage {
    static let root = URL.applicationSupportDirectory.appending(path: Bundle.main.bundleIdentifier ?? "DemoApp")

    static var modelsRoot: URL {
        let url = root.appending(path: "models")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        excludeFromBackup(url)
        return url
    }

    static func directory(for repo: String) -> URL {
        modelsRoot.appending(path: repo, directoryHint: .isDirectory)
    }

    static func key(_ url: URL) -> String {
        let path = url.standardizedFileURL.path(percentEncoded: false)
        return path.count > 1 && path.hasSuffix("/") ? String(path.dropLast()) : path
    }

    static func conversationKey(_ url: URL) -> String {
        ModelLocationIdentity.key(
            url, legacy: directory(for: "okayuji/Qwen3.8-27B-coreai-4bit"),
            managed: root.appending(path: "ModelLibrary/artifacts")
                .appending(path: LanguageModelArtifacts.qwenID)
                .appending(path: LanguageModelArtifacts.qwenRevision))
    }

    static var prepared: Set<String> {
        get { Set(UserDefaults.standard.stringArray(forKey: "preparedBundles") ?? []) }
        set { UserDefaults.standard.set(newValue.sorted(), forKey: "preparedBundles") }
    }

    static func size(of url: URL) -> Int64 {
        guard
            let walker = FileManager.default.enumerator(
                at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
        else {
            return 0
        }
        var total: Int64 = 0
        for case let file as URL in walker {
            guard let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
                values.isRegularFile == true
            else { continue }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }

    static func availableSpace() -> Int64 {
        let values = try? URL.homeDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }

    static func excludeFromBackup(_ url: URL) {
        var target = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? target.setResourceValues(values)
    }
}
