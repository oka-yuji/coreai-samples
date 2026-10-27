import ArgumentParser
import CryptoKit
import Darwin
import Foundation
import OCRCore
import OCRRuntime

enum OCRPromptKind: String, ExpressibleByArgument, Encodable {
    case text
    case table
    case layout

    var instruction: String {
        switch self {
        case .text: OCRPrompt.text
        case .table: OCRPrompt.table
        case .layout: OCRPrompt.layout
        }
    }
}

struct OCRBlock: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ocr-block",
        abstract: "Read one image block with the TeleOCR bundle on Core AI and print one JSON record per request."
    )

    @Argument(help: "TeleOCR bundle folder.")
    var bundle: String

    @Argument(help: "PNG or JPEG image, read as 8-bit RGB without alpha compositing or EXIF rotation.")
    var image: String

    @Option(help: "Prompt: text, table or layout.")
    var prompt: OCRPromptKind = .text

    @Option(help: "Crop box x0,y0,x1,y1 in source pixels, applied before the resize.")
    var crop: String?

    @Option(help: "Resize to N x N with Pillow bicubic after the crop.")
    var resize: Int?

    @Option(help: "Folder for text.txt, raw.txt and records.jsonl.")
    var out: String?

    @Option(help: "Maximum new tokens.")
    var maxNewTokens = OCRGeneration.defaultMaxNewTokens

    @Option(name: .customLong("repeat"), help: "Requests with the same input in this process.")
    var repeatCount = 1

    @Flag(help: "Delete the specialization cache entries of the three assets before loading.")
    var clearCache = false

    func validate() throws {
        guard repeatCount > 0, maxNewTokens > 0, (resize ?? 1) > 0 else {
            throw ValidationError("--repeat, --max-new-tokens and --resize take positive values.")
        }
        _ = try box()
    }

    func box() throws -> OCRBox? {
        guard let crop else { return nil }
        let values = crop.split(separator: ",", omittingEmptySubsequences: false).map { Int($0) }
        guard values.count == 4, let x0 = values[0], let y0 = values[1], let x1 = values[2], let y1 = values[3] else {
            throw ValidationError("--crop takes x0,y0,x1,y1.")
        }
        return OCRBox(x0: x0, y0: y0, x1: x1, y1: y1)
    }

    func run() async throws {
        let wiredBefore = wiredBytes()
        let bundleURL = URL(filePath: bundle)
        if clearCache { try OCRRuntime.deleteSpecializationCache(bundle: bundleURL) }
        let runtime = try await OCRRuntime.load(bundle: bundleURL)
        let file = try Data(contentsOf: URL(filePath: image))
        var source = try OCRImage(data: file)
        let box = try box()
        if let box { source = try source.cropped(to: box) }
        if let resize { source = try source.resized(width: resize, height: resize, resampling: .pillow) }
        let processor = try OCRImage.smartResize(width: source.width, height: source.height)
        let executable = try Bundle.main.executableURL.map { try hex(Data(SHA256.hash(data: Data(contentsOf: $0)))) }
        var lines = ""
        for request in 1...repeatCount {
            let result = try await runtime.recognize(
                source, instruction: prompt.instruction, maxNewTokens: maxNewTokens)
            var usage = rusage()
            let peak: Int? = getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : nil
            let decodeSeconds = result.totalSeconds - result.firstTokenSeconds
            let record = OCRBlockRecord(
                request: request, bundle: bundle, image: image, imageSha256: hex(Data(SHA256.hash(data: file))),
                prompt: prompt, crop: box.map { [$0.x0, $0.y0, $0.x1, $0.y1] }, resize: resize,
                imagePx: [source.width, source.height], processorPx: [processor.width, processor.height],
                grid: [result.gridHeight, result.gridWidth], windows: result.windowCount,
                imageTokens: result.imageTokens, promptTokens: result.promptTokens, outputTokens: result.tokens.count,
                finishedBy: result.finishedBy, text: result.text,
                rawText: result.rawText, tokens: result.tokens, loadS: runtime.loadStats.totalSeconds,
                tokenizerS: runtime.loadStats.tokenizerSeconds,
                assets: runtime.loadStats.assets.map { OCRAssetRecord(bundle: bundle, $0) },
                visionS: result.visionSeconds, firstTokenS: result.firstTokenSeconds, decodeS: decodeSeconds,
                generateS: result.totalSeconds, tokS: Double(result.tokens.count) / result.totalSeconds,
                decodeTokS: result.tokens.count > 1 ? Double(result.tokens.count - 1) / decodeSeconds : nil,
                memory: .now(), peakRssBytes: peak, footprintPeakBytes: footprintPeak(),
                pageins: getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_majflt : nil,
                wiredDeltaBytes: wiredBytes().flatMap { now in wiredBefore.map { Int64(now) - Int64($0) } },
                os: ProcessInfo.processInfo.operatingSystemVersionString, executableSha256: executable)
            try emit(record)
            lines += String(decoding: try JSONEncoder.records.encode(record), as: UTF8.self) + "\n"
            if request == repeatCount, let out {
                let folder = URL(filePath: out)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                try Data((result.text + "\n").utf8).write(to: folder.appending(path: "text.txt"))
                try Data(result.rawText.utf8).write(to: folder.appending(path: "raw.txt"))
                try Data(lines.utf8).write(to: folder.appending(path: "records.jsonl"))
            }
        }
    }
}

struct OCRAssetRecord: Encodable {
    let name: String
    let cached: Bool
    let seconds: Double
    let mainHash: String?

    init(bundle: String, _ asset: OCRLoadStats.Asset) {
        name = asset.name
        cached = asset.cached
        seconds = asset.seconds
        let url = URL(filePath: bundle).appending(path: asset.name + ".aimodel/main.hash")
        mainHash = (try? Data(contentsOf: url)).map(hex)
    }
}

extension OCRRecognition {
    var finishedBy: String { finish == .stopToken ? "eos" : "max_new_tokens" }
}

struct OCRBlockRecord: Encodable {
    var record = "ocr-block"
    let request: Int
    let bundle: String
    let image: String
    let imageSha256: String
    let prompt: OCRPromptKind
    let crop: [Int]?
    let resize: Int?
    let imagePx: [Int]
    let processorPx: [Int]
    let grid: [Int]
    let windows: Int
    let imageTokens: Int
    let promptTokens: Int
    let outputTokens: Int
    let finishedBy: String
    let text: String
    let rawText: String
    let tokens: [Int32]
    let loadS: Double
    let tokenizerS: Double
    let assets: [OCRAssetRecord]
    let visionS: Double
    let firstTokenS: Double
    let decodeS: Double
    let generateS: Double
    let tokS: Double
    let decodeTokS: Double?
    let memory: Memory
    let peakRssBytes: Int?
    let footprintPeakBytes: UInt64?
    let pageins: Int?
    let wiredDeltaBytes: Int64?
    let os: String
    let executableSha256: String?
}

func hex(_ data: Data) -> String {
    data.map { String(format: "%02x", $0) }.joined()
}

func footprintPeak() -> UInt64? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? UInt64(info.ledger_phys_footprint_peak) : nil
}

func wiredBytes() -> UInt64? {
    var stats = vm_statistics64_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &stats) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
        }
    }
    return result == KERN_SUCCESS ? UInt64(stats.wire_count) * UInt64(getpagesize()) : nil
}
