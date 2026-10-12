import CoreAIBackend
import Foundation
import Observation

@MainActor
@Observable
final class CatalogRow: Identifiable {
    let model: CatalogModel
    var isDownloading = false
    var progress = 0.0
    var downloadedBytes: Int64 = 0
    var totalBytes: Int64 = 0
    var bytesPerSecond = 0.0
    var error: String?
    var isDownloaded = false
    var hasPartial = false
    var diskSize: Int64 = 0

    @ObservationIgnored var downloader: HubDownloader?
    @ObservationIgnored var task: Task<Void, Never>?

    init(model: CatalogModel) {
        self.model = model
    }

    nonisolated var id: String { model.id }
    var directory: URL { ModelStorage.directory(for: model.repo) }

    func apply(_ progress: DownloadProgress) {
        self.progress = progress.fraction
        downloadedBytes = progress.downloadedBytes
        totalBytes = progress.totalBytes
        bytesPerSecond = progress.bytesPerSecond
    }
}

@MainActor
@Observable
final class ModelsViewModel {
    struct Folder: Identifiable {
        let url: URL
        let name: String?
        let size: Int64
        var problem: String?

        var id: String { ModelStorage.key(url) }
        var canLoad: Bool { name != nil && problem == nil }
        var canClear: Bool { name != nil }
    }

    struct MenuEntry: Identifiable {
        let name: String
        let url: URL
        let isCurrent: Bool

        var id: String { ModelStorage.key(url) }
    }

    static let foldersDefaultsKey = "localFolders"

    let rows: [CatalogRow]
    private(set) var folders: [Folder] = []
    private(set) var downloadedSize: Int64 = 0
    private(set) var availableSpace: Int64 = 0
    var message: String?
    @ObservationIgnored private let defaults: UserDefaults

    init(catalog: [CatalogModel] = Catalog.models, defaults: UserDefaults = .standard) {
        self.defaults = defaults
        rows = catalog.map { CatalogRow(model: $0) }
        refresh()
    }

    private var storedFolders: [String] {
        get { defaults.stringArray(forKey: Self.foldersDefaultsKey) ?? [] }
        set { defaults.set(newValue, forKey: Self.foldersDefaultsKey) }
    }

    func refresh() {
        let fileManager = FileManager.default
        for row in rows {
            row.isDownloaded = ModelStorage.isComplete(row.directory)
            row.hasPartial =
                !row.isDownloaded && fileManager.fileExists(atPath: row.directory.path(percentEncoded: false))
            row.diskSize = row.isDownloaded ? ModelStorage.size(of: row.directory) : 0
        }
        folders = storedFolders.map { path in
            let url = URL(filePath: path, directoryHint: .isDirectory)
            do {
                let name = try BundleSupport.supportedBundle(at: url).name
                return Folder(url: url, name: name, size: ModelStorage.size(of: url))
            } catch let refusal as BundleSupport.Refusal {
                return Folder(
                    url: url, name: try? ModelBundle(url: url).name, size: ModelStorage.size(of: url),
                    problem: refusal.reason)
            } catch {
                return Folder(
                    url: url, name: nil, size: ModelStorage.size(of: url), problem: error.localizedDescription)
            }
        }
        downloadedSize = ModelStorage.size(of: ModelStorage.modelsRoot)
        availableSpace = ModelStorage.availableSpace()
    }

    func menuEntries(current: URL?) -> (downloaded: [MenuEntry], onThisMac: [MenuEntry]) {
        let currentKey = current.map(ModelStorage.key)
        func entry(_ name: String, _ url: URL) -> MenuEntry {
            MenuEntry(name: name, url: url, isCurrent: ModelStorage.key(url) == currentKey)
        }
        return (
            rows.filter(\.isDownloaded).map { entry($0.model.name, $0.directory) },
            folders.filter(\.canLoad).compactMap { folder in folder.name.map { entry($0, folder.url) } }
        )
    }

    func open(_ url: URL) {
        do {
            _ = try BundleSupport.supportedBundle(at: url)
        } catch let refusal as BundleSupport.Refusal {
            message = refusal.reason
            return
        } catch {
            message = "\(url.lastPathComponent) is not a Core AI model bundle: \(error.localizedDescription)"
            return
        }
        let key = ModelStorage.key(url)
        if !storedFolders.contains(key) { storedFolders.append(key) }
        message = nil
        refresh()
    }

    func forget(_ folder: Folder) {
        storedFolders.removeAll { $0 == folder.id }
        refresh()
    }

    func clearPrepared(_ url: URL) {
        do {
            try Engine.deleteSpecializationCache(bundle: ModelBundle(url: url))
            ModelStorage.prepared.remove(ModelStorage.key(url))
            message = "Removed the prepared copy of \(url.lastPathComponent). The next load prepares it again."
        } catch {
            message = "Could not remove the prepared copy of \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }

    func startDownload(_ row: CatalogRow) {
        guard !row.isDownloading else { return }
        row.error = nil
        row.isDownloading = true
        row.progress = 0
        row.downloadedBytes = 0
        row.bytesPerSecond = 0
        row.totalBytes = row.model.sizeBytes
        let downloader = HubDownloader()
        row.downloader = downloader
        let model = row.model
        let destination = row.directory
        row.task = Task { [weak self] in
            do {
                try await downloader.download(repoID: model.repo, revision: model.revision, to: destination) {
                    progress in
                    Task { @MainActor in row.apply(progress) }
                }
                self?.finishDownload(row, error: nil)
            } catch {
                self?.finishDownload(row, error: error)
            }
        }
    }

    func cancelDownload(_ row: CatalogRow) {
        row.downloader?.cancel()
        row.task?.cancel()
        row.isDownloading = false
        refresh()
    }

    func delete(_ row: CatalogRow) {
        try? Engine.deleteSpecializationCache(bundle: ModelBundle(url: row.directory))
        ModelStorage.prepared.remove(ModelStorage.key(row.directory))
        try? FileManager.default.removeItem(at: row.directory)
        refresh()
    }

    private func finishDownload(_ row: CatalogRow, error: (any Error)?) {
        row.isDownloading = false
        row.downloader = nil
        if let error, !Self.isCancellation(error) {
            row.error = error.localizedDescription
        }
        refresh()
    }

    private static func isCancellation(_ error: any Error) -> Bool {
        if error is CancellationError { return true }
        if case HubDownloadError.cancelled = error { return true }
        return (error as NSError).code == NSURLErrorCancelled
    }
}
