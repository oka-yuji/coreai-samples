import Darwin
import Foundation

enum MetricsLog {
    struct Record: Encodable {
        let timestamp: String
        let appBuild: String
        let osVersion: String
        let deviceModel: String
        let model: String
        let turn: Int
        let inputTokens: Int?
        let cachedTokens: Int?
        let outputTokens: Int
        let ttftSeconds: Double?
        let tokensPerSecond: Double?
        let stopReason: String
        let error: String?
        let rssMB: Double
    }

    static var fileURL: URL { ModelStorage.root.appending(path: "metrics.jsonl") }

    private static let queue = DispatchQueue(label: "com.coreai-samples.DemoApp.metrics")

    static func generation(model: String, stats: GenerationStats) {
        let record = Record(
            timestamp: Date.now.ISO8601Format(),
            appBuild: appBuild(),
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            deviceModel: deviceModel(),
            model: model,
            turn: stats.turn,
            inputTokens: stats.inputTokens,
            cachedTokens: stats.cachedTokens,
            outputTokens: stats.outputTokens,
            ttftSeconds: stats.ttftSeconds,
            tokensPerSecond: stats.tokensPerSecond,
            stopReason: stats.stopReason,
            error: stats.error,
            rssMB: Double(residentBytes()) / 1_048_576)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard var line = try? encoder.encode(record) else { return }
        line.append(0x0A)
        let data = line
        let url = fileURL
        queue.async { append(data, to: url) }
    }

    static func flush() async {
        await withCheckedContinuation { continuation in
            queue.async { continuation.resume() }
        }
    }

    private static func append(_ data: Data, to url: URL) {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let descriptor = open(url.path(percentEncoded: false), O_WRONLY | O_APPEND | O_CREAT, 0o644)
        guard descriptor >= 0 else { return }
        _ = data.withUnsafeBytes { write(descriptor, $0.baseAddress, $0.count) }
        close(descriptor)
    }

    private static func residentBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.resident_size : 0
    }

    private static func deviceModel() -> String {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var value = [UInt8](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.model", &value, &size, nil, 0)
        return String(decoding: value.prefix { $0 != 0 }, as: UTF8.self)
    }

    private static func appBuild() -> String {
        let info = Bundle.main.infoDictionary
        return "\(info?["CFBundleShortVersionString"] as? String ?? "?") (\(info?["CFBundleVersion"] as? String ?? "?"))"
    }
}
