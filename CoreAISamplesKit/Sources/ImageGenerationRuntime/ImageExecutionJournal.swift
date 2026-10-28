import Foundation
import ImageGenerationCore

protocol ImageCheckpointSink: Actor {
    func record(
        _ progress: ImageExecutionProgress, report: ImageExecutionReport, memory: ImageMemoryObservation
    ) async throws
}

actor ImageExecutionJournal: ImageCheckpointSink {
    private struct Checkpoint: Encodable {
        let format = 1
        let sequence: Int
        let processID: Int32
        let uptimeSeconds: Double
        let phase: String
        let component: ImageModelComponent?
        let step: Int?
        let memory: ImageMemoryObservation
        let report: ImageExecutionReport
    }

    private let directory: URL
    private var identity: UUID?
    private var sequence = 0

    init(directory: URL) { self.directory = directory }

    func record(
        _ progress: ImageExecutionProgress, report: ImageExecutionReport, memory: ImageMemoryObservation
    ) throws {
        guard directory.isFileURL, progress.requestID == report.requestID, report.status == .running,
            identity == nil || identity == report.requestID
        else { throw ImageExecutionError.invalidResource }
        let values = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true else {
            throw ImageExecutionError.invalidResource
        }
        let destination = directory.appending(path: "checkpoint.json")
        if sequence == 0 {
            guard !FileManager.default.fileExists(atPath: destination.path) else {
                throw ImageExecutionError.invalidResource
            }
        } else {
            let existing = try destination.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard existing.isRegularFile == true, existing.isSymbolicLink != true else {
                throw ImageExecutionError.invalidResource
            }
        }
        let phase: String
        let component: ImageModelComponent?
        let step: Int?
        switch progress.phase {
        case .preparing: (phase, component, step) = ("preparing", nil, nil)
        case .loading(let kind): (phase, component, step) = ("loading", kind, nil)
        case .running(let kind, let index): (phase, component, step) = ("inference", kind, index)
        case .releasing(let kind): (phase, component, step) = ("release", kind, nil)
        }
        let checkpoint = Checkpoint(
            sequence: sequence + 1, processID: ProcessInfo.processInfo.processIdentifier,
            uptimeSeconds: ProcessInfo.processInfo.systemUptime, phase: phase, component: component, step: step,
            memory: memory, report: report)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(checkpoint).write(to: destination, options: .atomic)
        identity = report.requestID
        sequence += 1
    }
}
