import CryptoKit
import Foundation

struct HubFile: Sendable {
    let path: String
    let size: Int64
    let checksum: Checksum

    enum Checksum: Sendable {
        case sha256(String)
        case gitBlobSHA1(String)

        var value: String {
            switch self {
            case .sha256(let value), .gitBlobSHA1(let value): value
            }
        }
    }
}

struct DownloadProgress: Sendable {
    let fraction: Double
    let downloadedBytes: Int64
    let totalBytes: Int64
    let bytesPerSecond: Double
}

enum HubDownloadError: LocalizedError {
    case invalidURL(String)
    case httpError(statusCode: Int, url: String)
    case invalidResponse(url: String)
    case cancelled
    case invalidRevision(String)
    case invalidFile(String)
    case checksumMismatch(String)
    case insufficientDiskSpace(required: Int64, available: Int64)

    var errorDescription: String? {
        switch self {
        case .invalidURL(let url):
            "Invalid URL: \(url)"
        case .httpError(let code, let url):
            "HTTP \(code) for \(url)"
        case .invalidResponse(let url):
            "Invalid response from \(url)"
        case .invalidRevision(let revision):
            "A 40-character commit hash is required, not '\(revision)'."
        case .invalidFile(let path):
            "Missing or invalid size, path, or checksum for '\(path)'."
        case .checksumMismatch(let path):
            "Downloaded file failed its size or checksum check: \(path)"
        case .cancelled:
            "Download cancelled"
        case .insufficientDiskSpace(let required, let available):
            "Insufficient disk space. Required \(required.formatted(.byteCount(style: .file))), available \(available.formatted(.byteCount(style: .file)))"
        }
    }
}

final class HubDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let baseURL = "https://huggingface.co"
    private let lock = NSLock()
    private var entries: [Int: Entry] = [:]
    private var liveTasks: [Int: URLSessionDownloadTask] = [:]
    private var didCancel = false
    private var storedSession: URLSession?
    private let listLoader: (@Sendable (URL) async throws -> (Data, URLResponse))?
    private let fileLoader: (@Sendable (URL, URL) async throws -> URLResponse)?
    private let maximumConcurrentDownloads: Int

    init(
        listLoader: (@Sendable (URL) async throws -> (Data, URLResponse))? = nil,
        fileLoader: (@Sendable (URL, URL) async throws -> URLResponse)? = nil,
        maximumConcurrentDownloads: Int = 6
    ) {
        self.listLoader = listLoader
        self.fileLoader = fileLoader
        self.maximumConcurrentDownloads = max(1, maximumConcurrentDownloads)
        super.init()
    }

    private struct Entry {
        let destinationFile: URL
        let tracker: ProgressTracker
        let progressHandler: (@Sendable (DownloadProgress) -> Void)?
        let continuation: CheckedContinuation<URLResponse, Error>
        var lastBytes: Int64 = 0
        var didResume = false
    }

    private func session() throws -> URLSession {
        try lock.withLock {
            guard !didCancel else { throw HubDownloadError.cancelled }
            if let storedSession { return storedSession }
            let config = URLSessionConfiguration.default
            config.waitsForConnectivity = true
            config.timeoutIntervalForRequest = 120
            config.timeoutIntervalForResource = 3600
            let result = URLSession(configuration: config, delegate: self, delegateQueue: nil)
            storedSession = result
            return result
        }
    }

    func fileList(repoID: String, revision: String) async throws -> [HubFile] {
        guard revision.utf8.count == 40, revision.utf8.allSatisfy(Self.isHexDigit) else {
            throw HubDownloadError.invalidRevision(revision)
        }
        let urlString = "\(baseURL)/api/models/\(repoID)/tree/\(revision)?recursive=true"
        guard let url = URL(string: urlString) else { throw HubDownloadError.invalidURL(urlString) }
        try checkCancellation()
        let data: Data
        let response: URLResponse
        if let listLoader {
            (data, response) = try await listLoader(url)
        } else {
            (data, response) = try await session().data(for: URLRequest(url: url))
        }
        try checkCancellation()
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw HubDownloadError.httpError(
                statusCode: (response as? HTTPURLResponse)?.statusCode ?? -1, url: urlString)
        }
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [[String: Any]],
            http.value(forHTTPHeaderField: "Link")?.contains("rel=\"next\"") != true
        else { throw HubDownloadError.invalidResponse(url: urlString) }
        var result: [HubFile] = []
        var paths: Set<String> = []
        for entry in raw where entry["type"] as? String == "file" {
            let path = entry["path"] as? String ?? "<unknown>"
            let parts = path.split(separator: "/", omittingEmptySubsequences: false)
            guard !parts.isEmpty, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
                !path.contains("\0"), !path.contains("\\"), path != ModelStorage.markerName,
                paths.insert(path).inserted, let size = entry["size"] as? Int64, size >= 0
            else { throw HubDownloadError.invalidFile(path) }
            let checksum: HubFile.Checksum
            if let lfs = entry["lfs"] as? [String: Any] {
                guard let value = lfs["oid"] as? String, value.utf8.count == 64,
                    value.utf8.allSatisfy(Self.isHexDigit), lfs["size"] as? Int64 == size
                else { throw HubDownloadError.invalidFile(path) }
                checksum = .sha256(value.lowercased())
            } else {
                guard entry["lfs"] == nil, let value = entry["oid"] as? String,
                    value.utf8.count == 40, value.utf8.allSatisfy(Self.isHexDigit)
                else { throw HubDownloadError.invalidFile(path) }
                checksum = .gitBlobSHA1(value.lowercased())
            }
            result.append(HubFile(path: path, size: size, checksum: checksum))
        }
        guard !result.isEmpty else { throw HubDownloadError.invalidResponse(url: urlString) }
        return result
    }

    func download(
        repoID: String,
        revision: String,
        to destinationDir: URL,
        progress: (@Sendable (DownloadProgress) -> Void)? = nil,
        verified: (@Sendable (String, Bool) -> Void)? = nil
    ) async throws {
        defer { lock.withLock { storedSession }?.finishTasksAndInvalidate() }
        try await withTaskCancellationHandler {
            try await performDownload(
                repoID: repoID, revision: revision, to: destinationDir, progress: progress, verified: verified)
        } onCancel: {
            self.cancel()
        }
    }

    private func performDownload(
        repoID: String, revision: String, to destinationDir: URL,
        progress: (@Sendable (DownloadProgress) -> Void)?, verified: (@Sendable (String, Bool) -> Void)?
    ) async throws {
        let files = try await fileList(repoID: repoID, revision: revision)
        let totalBytes = try files.reduce(Int64(0)) { partial, file in
            let (sum, overflow) = partial.addingReportingOverflow(file.size)
            guard !overflow else { throw HubDownloadError.invalidFile(file.path) }
            return sum
        }
        try checkCancellation()
        try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)
        let marker = destinationDir.appending(path: ModelStorage.markerName)
        try lock.withLock {
            guard !didCancel, !Task.isCancelled else { throw HubDownloadError.cancelled }
            if FileManager.default.fileExists(atPath: marker.path(percentEncoded: false)) {
                try FileManager.default.removeItem(at: marker)
            }
        }
        let tracker = ProgressTracker(totalBytes: totalBytes)
        var pending: [HubFile] = []
        var alreadyPresent: Int64 = 0
        for file in files {
            try checkCancellation()
            let destination = try destination(for: file, in: destinationDir)
            if try await matches(file, at: destination) {
                alreadyPresent += file.size
                report(tracker.add(bytes: file.size), tracker: tracker, progress: progress)
                verified?(file.path, true)
            } else {
                try lock.withLock {
                    guard !didCancel, !Task.isCancelled else { throw HubDownloadError.cancelled }
                    if FileManager.default.fileExists(atPath: destination.path(percentEncoded: false)) {
                        try FileManager.default.removeItem(at: destination)
                    }
                }
                pending.append(file)
            }
        }
        try checkCancellation()
        let needed = totalBytes - alreadyPresent
        let available = ModelStorage.availableSpace()
        if available > 0, available < needed {
            throw HubDownloadError.insufficientDiskSpace(required: needed, available: available)
        }
        try await withThrowingTaskGroup(of: String.self) { group in
            var iterator = pending.makeIterator()
            for _ in 0..<min(maximumConcurrentDownloads, pending.count) {
                if let file = iterator.next() {
                    group.addTask { [self] in
                        try await downloadFile(
                            file: file, repoID: repoID, revision: revision,
                            destinationDir: destinationDir, tracker: tracker, progress: progress)
                        return file.path
                    }
                }
            }
            while let path = try await group.next() {
                verified?(path, false)
                try checkCancellation()
                if let file = iterator.next() {
                    group.addTask { [self] in
                        try await downloadFile(
                            file: file, repoID: repoID, revision: revision,
                            destinationDir: destinationDir, tracker: tracker, progress: progress)
                        return file.path
                    }
                }
            }
        }
        let completion = ModelStorage.CompletionMarker(
            fileCount: files.count, totalBytes: totalBytes, completedAt: Date.now.ISO8601Format())
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let markerData = try encoder.encode(completion)
        try lock.withLock {
            guard !didCancel, !Task.isCancelled else { throw HubDownloadError.cancelled }
            try markerData.write(to: marker, options: .atomic)
        }
    }

    func cancel() {
        let (tasks, session) = lock.withLock {
            didCancel = true
            return (Array(liveTasks.values), storedSession)
        }
        for task in tasks { task.cancel() }
        session?.invalidateAndCancel()
    }

    private func downloadFile(
        file: HubFile, repoID: String, revision: String, destinationDir: URL,
        tracker: ProgressTracker, progress: (@Sendable (DownloadProgress) -> Void)?
    ) async throws {
        try checkCancellation()
        let destination = try destination(for: file, in: destinationDir)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        let temporary = destination.deletingLastPathComponent().appending(path: ".\(UUID().uuidString).partial")
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard let base = URL(string: "\(baseURL)/\(repoID)/resolve/\(revision)") else {
            throw HubDownloadError.invalidURL(repoID)
        }
        let url = base.appending(path: file.path)
        let response: URLResponse
        if let fileLoader {
            response = try await fileLoader(url, temporary)
            report(tracker.add(bytes: file.size), tracker: tracker, progress: progress)
        } else {
            response = try await downloadOne(url: url, destinationFile: temporary, tracker: tracker, progress: progress)
        }
        try checkCancellation()
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw HubDownloadError.httpError(
                statusCode: (response as? HTTPURLResponse)?.statusCode ?? -1, url: url.absoluteString)
        }
        guard try await matches(file, at: temporary) else { throw HubDownloadError.checksumMismatch(file.path) }
        try lock.withLock {
            guard !didCancel, !Task.isCancelled else { throw HubDownloadError.cancelled }
            _ = try self.destination(for: file, in: destinationDir)
            try FileManager.default.moveItem(at: temporary, to: destination)
        }
    }

    private func destination(for file: HubFile, in directory: URL) throws -> URL {
        let root = directory.resolvingSymlinksInPath()
        let result = root.appending(path: file.path)
        let components = result.resolvingSymlinksInPath().pathComponents
        guard components.count > root.pathComponents.count,
            components.starts(with: root.pathComponents)
        else { throw HubDownloadError.invalidFile(file.path) }
        return result
    }

    private func matches(_ file: HubFile, at url: URL) async throws -> Bool {
        try checkCancellation()
        guard fileSize(at: url) == file.size else { return false }
        let digest = try await Self.fileDigest(at: url, checksum: file.checksum, size: file.size) { [self] in
            isCancelled()
        }
        try checkCancellation()
        return digest == file.checksum.value
    }

    @concurrent
    static func fileDigest(
        at url: URL, checksum: HubFile.Checksum, size: Int64, cancelled: @Sendable () -> Bool = { false }
    ) async throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var sha256 = SHA256()
        var sha1 = Insecure.SHA1()
        if case .gitBlobSHA1 = checksum { sha1.update(data: Data("blob \(size)\0".utf8)) }
        while true {
            guard !Task.isCancelled, !cancelled() else { throw HubDownloadError.cancelled }
            guard let data = try handle.read(upToCount: 1_048_576), !data.isEmpty else { break }
            switch checksum {
            case .sha256: sha256.update(data: data)
            case .gitBlobSHA1: sha1.update(data: data)
            }
        }
        switch checksum {
        case .sha256: return sha256.finalize().map { String(format: "%02x", $0) }.joined()
        case .gitBlobSHA1: return sha1.finalize().map { String(format: "%02x", $0) }.joined()
        }
    }

    private static func isHexDigit(_ byte: UInt8) -> Bool {
        (48...57).contains(byte) || (65...70).contains(byte) || (97...102).contains(byte)
    }

    private func checkCancellation() throws {
        guard !Task.isCancelled, !isCancelled() else { throw HubDownloadError.cancelled }
    }

    private func report(
        _ result: ProgressTracker.Result, tracker: ProgressTracker, progress: (@Sendable (DownloadProgress) -> Void)?
    ) {
        if result.shouldReport {
            progress?(
                DownloadProgress(
                    fraction: result.fraction, downloadedBytes: result.completed,
                    totalBytes: tracker.totalBytes, bytesPerSecond: result.bytesPerSecond ?? 0))
        }
    }

    private func downloadOne(
        url: URL,
        destinationFile: URL,
        tracker: ProgressTracker,
        progress: (@Sendable (DownloadProgress) -> Void)?
    ) async throws -> URLResponse {
        let session = try session()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<URLResponse, Error>) in
                lock.lock()
                if didCancel {
                    lock.unlock()
                    continuation.resume(throwing: HubDownloadError.cancelled)
                    return
                }
                let task = session.downloadTask(with: URLRequest(url: url))
                let id = task.taskIdentifier
                entries[id] = Entry(
                    destinationFile: destinationFile,
                    tracker: tracker,
                    progressHandler: progress,
                    continuation: continuation
                )
                liveTasks[id] = task
                lock.unlock()
                task.resume()
            }
        } onCancel: {
            self.cancel()
        }
    }

    private func isCancelled() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return didCancel
    }

    private func fileSize(at url: URL) -> Int64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path(percentEncoded: false)) else {
            return nil
        }
        return (attributes[.size] as? NSNumber)?.int64Value
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let id = downloadTask.taskIdentifier
        lock.lock()
        guard var entry = entries[id] else {
            lock.unlock()
            return
        }
        let delta = totalBytesWritten - entry.lastBytes
        entry.lastBytes = totalBytesWritten
        let tracker = entry.tracker
        let handler = entry.progressHandler
        entries[id] = entry
        lock.unlock()

        guard delta > 0 else { return }
        let result = tracker.add(bytes: delta)
        if result.shouldReport {
            handler?(
                DownloadProgress(
                    fraction: result.fraction,
                    downloadedBytes: result.completed,
                    totalBytes: tracker.totalBytes,
                    bytesPerSecond: result.bytesPerSecond ?? 0
                ))
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        let id = downloadTask.taskIdentifier
        lock.lock()
        guard let entry = entries[id], !entry.didResume else {
            lock.unlock()
            return
        }
        let destination = entry.destinationFile
        lock.unlock()

        let moveError: Error?
        do {
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if FileManager.default.fileExists(atPath: destination.path(percentEncoded: false)) {
                try? FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: location, to: destination)
            moveError = nil
        } catch {
            moveError = error
        }

        lock.lock()
        liveTasks.removeValue(forKey: id)
        guard var entry = entries[id], !entry.didResume else {
            lock.unlock()
            return
        }
        entry.didResume = true
        entries[id] = entry
        let continuation = entry.continuation
        lock.unlock()

        if let moveError {
            continuation.resume(throwing: moveError)
        } else if let response = downloadTask.response {
            continuation.resume(returning: response)
        } else {
            try? FileManager.default.removeItem(at: destination)
            continuation.resume(
                throwing: HubDownloadError.invalidResponse(url: downloadTask.originalRequest?.url?.absoluteString ?? "")
            )
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        let id = task.taskIdentifier
        lock.lock()
        liveTasks.removeValue(forKey: id)
        guard var entry = entries[id], !entry.didResume else {
            entries.removeValue(forKey: id)
            lock.unlock()
            return
        }
        entry.didResume = true
        entries[id] = entry
        let continuation = entry.continuation
        lock.unlock()

        if let error {
            if (error as NSError).code == NSURLErrorCancelled {
                continuation.resume(throwing: HubDownloadError.cancelled)
            } else {
                continuation.resume(throwing: error)
            }
        } else {
            continuation.resume(
                throwing: HubDownloadError.invalidResponse(url: task.originalRequest?.url?.absoluteString ?? ""))
        }

        lock.lock()
        entries.removeValue(forKey: id)
        lock.unlock()
    }
}

final class ProgressTracker: @unchecked Sendable {
    let totalBytes: Int64
    private let startTime = Date()
    private let lock = NSLock()
    private var completedBytes: Int64 = 0
    private var lastReportedAt = Date.distantPast

    struct Result {
        let fraction: Double
        let completed: Int64
        let bytesPerSecond: Double?
        let shouldReport: Bool
    }

    init(totalBytes: Int64) {
        self.totalBytes = totalBytes
    }

    func add(bytes: Int64) -> Result {
        let now = Date()
        lock.lock()
        completedBytes += bytes
        let isComplete = totalBytes > 0 && completedBytes >= totalBytes
        let shouldReport = now.timeIntervalSince(lastReportedAt) >= 0.1 || isComplete
        if shouldReport { lastReportedAt = now }
        let completed = completedBytes
        lock.unlock()

        let fraction = totalBytes > 0 ? min(Double(completed) / Double(totalBytes), 1.0) : 0
        let elapsed = now.timeIntervalSince(startTime)
        let bytesPerSecond = elapsed > 0.5 ? Double(completed) / elapsed : nil
        return Result(
            fraction: fraction, completed: completed, bytesPerSecond: bytesPerSecond, shouldReport: shouldReport)
    }
}
