import CryptoKit
import ForecastFeature
import Foundation

public enum SalesSource {
    public static let url =
        "https://data.mendeley.com/public-files/datasets/njdkntcpc9/files/08bb4f43-6dfa-4995-b268-42fa0690ba6b/file_downloaded"
    public static let bytes = 888_985
    public static let sha256 = "0dfd4a5bf801bf79ddb5fb1f6bc8d023487c36dd6a2a70b312cdfa9fa6568d83"
    public static let byteLimit = 2 * 1024 * 1024

    public static func validate(_ data: Data) throws {
        guard data.count <= byteLimit else { throw ForecastFailure.responseTooLarge }
        guard data.count == bytes,
            SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == sha256
        else { throw ForecastFailure.sourceMismatch }
    }
}

public actor SalesDataRepository {
    public typealias Fetch = @Sendable () async throws -> Data
    private let cacheDirectory: URL?
    private let fetch: Fetch
    private let admit: @Sendable (Data) throws -> ForecastDataset
    private var busy = false

    public init(cacheDirectory: URL? = nil, fetch: Fetch? = nil) {
        self.cacheDirectory =
            cacheDirectory
            ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
            .first?.appending(path: Bundle.main.bundleIdentifier ?? "CoreAISamples")
            .appending(path: "Forecast/Sales-v1")
        self.fetch = fetch ?? Self.download
        admit = { data in
            try SalesSource.validate(data)
            return try SalesDataset.parse(data)
        }
    }

    init(cacheDirectory: URL, fetch: @escaping Fetch, admit: @escaping @Sendable (Data) throws -> ForecastDataset) {
        self.cacheDirectory = cacheDirectory
        self.fetch = fetch
        self.admit = admit
    }

    public func load(refresh: Bool = false) async throws -> ForecastDataset {
        guard !busy else { throw ForecastFailure.busy }
        busy = true
        defer { busy = false }
        try Task.checkCancellation()
        guard let cacheDirectory else { throw ForecastFailure.cache }
        let cache = cacheDirectory.appending(path: "sales.csv")
        if !refresh, FileManager.default.fileExists(atPath: cache.path) {
            let data = try read(cache)
            let dataset = try admit(data)
            try Task.checkCancellation()
            return dataset
        }
        let data: Data
        do {
            data = try await fetch()
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as ForecastFailure {
            throw error
        } catch {
            throw ForecastFailure.network
        }
        try Task.checkCancellation()
        guard data.count <= SalesSource.byteLimit else { throw ForecastFailure.responseTooLarge }
        let dataset = try admit(data)
        try Task.checkCancellation()
        do {
            try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
            try data.write(to: cache, options: .atomic)
        } catch {
            throw ForecastFailure.cache
        }
        return dataset
    }

    private func read(_ url: URL) throws -> Data {
        do {
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true,
                let size = values.fileSize, size <= SalesSource.byteLimit
            else { throw ForecastFailure.sourceMismatch }
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            let data = try handle.read(upToCount: SalesSource.byteLimit + 1) ?? Data()
            guard data.count <= SalesSource.byteLimit else { throw ForecastFailure.responseTooLarge }
            return data
        } catch let error as ForecastFailure {
            throw error
        } catch {
            throw ForecastFailure.cache
        }
    }

    private static func download() async throws -> Data {
        guard let url = URL(string: SalesSource.url) else { throw ForecastFailure.invalidData }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.urlCredentialStorage = nil
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 120
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let (bytes, response) = try await session.bytes(from: url)
        guard let response = response as? HTTPURLResponse, response.statusCode == 200 else {
            throw ForecastFailure.network
        }
        guard response.expectedContentLength <= SalesSource.byteLimit else { throw ForecastFailure.responseTooLarge }
        var data = Data()
        for try await byte in bytes {
            try Task.checkCancellation()
            guard data.count < SalesSource.byteLimit else { throw ForecastFailure.responseTooLarge }
            data.append(byte)
        }
        return data
    }
}
