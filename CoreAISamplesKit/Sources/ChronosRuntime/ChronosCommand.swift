import ChronosCore
import Darwin
import Foundation

public enum ChronosCommand: Equatable, Sendable {
    public enum Mode: Equatable, Sendable {
        case input(URL)
        case verify
    }

    case help
    case run(bundle: URL, mode: Mode, output: URL)

    public static let usage = """
        chronos-verify --bundle <folder> --input <json> --output <new-file>
        chronos-verify --bundle <folder> --verify --output <new-file>
        chronos-verify --help
        Input JSON: {"context":[number|null,...]}. Output files are never overwritten.
        Verification compares all 18 cases and 6048 values, then repeats the first case after release/reload.
        Runtime timings include hashing and model loading only on requests marked loaded.
        """

    public static func parse(_ arguments: [String]) throws -> Self {
        if arguments == ["--help"] { return .help }
        var values: [String: String] = [:]
        var verify = false
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            if argument == "--verify" {
                guard !verify else { throw ChronosError.invalidArguments }
                verify = true
                index += 1
            } else {
                guard ["--bundle", "--input", "--output"].contains(argument), values[argument] == nil,
                    index + 1 < arguments.count, !arguments[index + 1].isEmpty,
                    !arguments[index + 1].hasPrefix("--")
                else { throw ChronosError.invalidArguments }
                values[argument] = arguments[index + 1]
                index += 2
            }
        }
        guard let bundle = values["--bundle"], let output = values["--output"], verify != (values["--input"] != nil)
        else {
            throw ChronosError.invalidArguments
        }
        let mode: Mode
        if let input = values["--input"] {
            mode = .input(try fileURL(input))
        } else {
            mode = .verify
        }
        return .run(
            bundle: try fileURL(bundle), mode: mode, output: try fileURL(output))
    }

    private static func fileURL(_ path: String) throws -> URL {
        let url = URL(fileURLWithPath: path)
        guard !path.contains("\0"), let parent = realpath(url.deletingLastPathComponent().path, nil) else {
            throw ChronosError.invalidArguments
        }
        defer { free(parent) }
        return URL(fileURLWithPath: String(cString: parent), isDirectory: true)
            .appendingPathComponent(url.lastPathComponent)
    }

    public static func decodeInput(_ data: Data) throws -> ChronosContext {
        struct Input: Decodable {
            let context: [Float?]
        }
        guard data.count <= ChronosFiles.inputLimit,
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any], Set(object.keys) == ["context"]
        else { throw ChronosError.invalidInput }
        let input = try ChronosJSON.decode(Input.self, from: data)
        return try ChronosContext(input.context.map { $0 ?? .nan })
    }

    @concurrent
    public static func execute(_ command: Self) async throws {
        try await execute(command, runtimeFactory: { ChronosRuntime(bundleURL: $0) })
    }

    @concurrent
    static func execute(_ command: Self, runtimeFactory: @Sendable (URL) -> ChronosRuntime) async throws {
        guard case .run(let bundleURL, let mode, let output) = command else { return }
        try ChronosReportFile.requireNew(output)
        let runtime = runtimeFactory(bundleURL)
        do {
            switch mode {
            case .input(let url):
                let context = try decodeInput(ChronosFiles.read(url, limit: ChronosFiles.inputLimit))
                let result = try await runtime.predict(context)
                await runtime.release()
                try Task.checkCancellation()
                try ChronosReportFile.save(ChronosPredictionReport(result: result), to: output)
            case .verify:
                let bundle = try await ChronosBundle.open(bundleURL)
                let report = try await ChronosVerification.run(bundle: bundle, runtime: runtime)
                try Task.checkCancellation()
                try ChronosReportFile.save(report, to: output)
                guard report.passed else { throw ChronosError.verificationFailed }
            }
        } catch {
            await runtime.release()
            throw error
        }
    }
}

struct ChronosPredictionReport: Encodable {
    let format = "coreai-samples-chronos2-swift-prediction/1"
    let precision = "float32"
    let shape = ChronosContract.outputShape
    let quantileLevels = ChronosContract.quantileLevels
    let identity: ChronosBundleIdentity
    let rawCount: Int
    let paddedCount: Int
    let quantiles: [[[Float]]]
    let seconds: Double
    let loaded: Bool
    let timingBoundary: String

    init(result: ChronosResult) {
        identity = result.identity
        rawCount = result.rawCount
        paddedCount = result.paddedCount
        quantiles = [result.prediction.quantiles]
        seconds = result.seconds
        loaded = result.loaded
        timingBoundary = result.timingBoundary
    }
}

public enum ChronosReportFile {
    public static func requireNew(_ url: URL) throws {
        let directory = try ChronosFiles.openDirectory(url.deletingLastPathComponent(), followFinalLink: true)
        defer { Darwin.close(directory) }
        try requireNew(url.lastPathComponent, in: directory)
    }

    public static func save(_ value: some Encodable, to url: URL) throws {
        let data = try ChronosJSON.encode(value)
        let directory = try ChronosFiles.openDirectory(url.deletingLastPathComponent(), followFinalLink: true)
        defer { Darwin.close(directory) }
        let name = url.lastPathComponent
        try requireNew(name, in: directory)
        let temporary = ".chronos-\(UUID().uuidString).tmp"
        let descriptor = openat(directory, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else { throw ChronosFiles.failure() }
        defer {
            Darwin.close(descriptor)
            unlinkat(directory, temporary, 0)
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        try handle.write(contentsOf: data)
        try handle.synchronize()
        try Task.checkCancellation()
        guard linkat(directory, temporary, directory, name, 0) == 0 else {
            if errno == EEXIST { throw ChronosError.outputExists }
            throw ChronosFiles.failure()
        }
    }

    private static func requireNew(_ name: String, in directory: Int32) throws {
        try ChronosFiles.validateRelativePath(name)
        var info = stat()
        if fstatat(directory, name, &info, AT_SYMLINK_NOFOLLOW) == 0 { throw ChronosError.outputExists }
        guard errno == ENOENT else { throw ChronosFiles.failure() }
    }
}
