import CryptoKit
import ForecastFeature
import Foundation

public enum BikeSource {
    public static let normalizedBytes = 604_168
    public static let normalizedSHA256 = "d2f7a0f0ba95e82678ead794429a18a7f696aa3b82c9b9ec34ec47806ff0767b"
    public static let originalSHA256 = "373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60"
    public static let manifestSHA256 = "b0dc815d7d5f97f2aa0d1aeafdf116b756f273b3e660aab73a1e07fd15966593"

    static var bundledDirectory: URL? { Bundle.module.url(forResource: "SeoulBike", withExtension: nil) }

    static func read(_ directory: URL) throws -> Data {
        let files = [
            ("SeoulBikeData.csv", normalizedBytes, normalizedSHA256),
            ("manifest.json", 2100, manifestSHA256),
            ("NOTICE.txt", 832, "64137efc03276045d14e77e9e5402fa2cccf49333852c0df2e5ba469fb5d9f27"),
        ]
        do {
            let folder = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard folder.isDirectory == true, folder.isSymbolicLink != true,
                Set(try FileManager.default.contentsOfDirectory(atPath: directory.path)) == Set(files.map(\.0))
            else { throw ForecastFailure.sourceMismatch }
            var source = Data()
            for (name, size, digest) in files {
                try Task.checkCancellation()
                let file = directory.appending(path: name)
                let info = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
                guard info.isRegularFile == true, info.isSymbolicLink != true, info.fileSize == size else {
                    throw ForecastFailure.sourceMismatch
                }
                let handle = try FileHandle(forReadingFrom: file)
                defer { try? handle.close() }
                let data = try handle.read(upToCount: size + 1) ?? Data()
                guard data.count == size,
                    SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == digest
                else { throw ForecastFailure.sourceMismatch }
                if name == "SeoulBikeData.csv" { source = data }
            }
            return source
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw ForecastFailure.sourceMismatch
        }
    }
}

public actor BikeDataRepository: ForecastDataLoading {
    private let directory: URL?

    public init(directory: URL? = nil) {
        self.directory = directory ?? BikeSource.bundledDirectory
    }

    public func load(refresh: Bool = false) async throws -> ForecastDataset {
        try Task.checkCancellation()
        guard let directory else { throw ForecastFailure.sourceMismatch }
        let data = try BikeSource.read(directory)
        let dataset = try BikeDataset.parse(data)
        try Task.checkCancellation()
        return dataset
    }
}
