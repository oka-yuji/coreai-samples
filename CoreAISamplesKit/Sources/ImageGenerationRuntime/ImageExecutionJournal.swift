import Foundation
import ImageGenerationCore

protocol ImageCheckpointSink: Actor {
    func record(
        _ progress: ImageExecutionProgress, report: ImageExecutionReport, memory: ImageMemoryObservation
    ) async throws
}

struct ImageNativeTensorObservation: Encodable, Sendable {
    let name: String
    let shape: [Int]
    let strides: [Int]
    let minimumByteCount: Int
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

    private struct NativeCheckpoint: Encodable {
        let format = 1
        let sequence: Int
        let checkpointSequence: Int
        let processID: Int32
        let uptimeSeconds: Double
        let requestID: UUID
        let profileID: String
        let manifestSHA256: String
        let component: ImageModelComponent?
        let step: Int?
        let asset: String
        let stage: String
        let tensor: ImageNativeTensorObservation?
        let memory: ImageMemoryObservation
    }

    private let directory: URL
    private var identity: UUID?
    private var sequence = 0
    private var nativeSequence = 0
    private var latest: Checkpoint?

    init(directory: URL) { self.directory = directory }

    func record(
        _ progress: ImageExecutionProgress, report: ImageExecutionReport, memory: ImageMemoryObservation
    ) throws {
        guard directory.isFileURL, progress.requestID == report.requestID, report.status == .running,
            identity == nil || identity == report.requestID
        else { throw ImageExecutionError.invalidResource }
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
        try write(checkpoint, name: "checkpoint.json", replacing: sequence > 0)
        identity = report.requestID
        latest = checkpoint
        sequence += 1
    }

    func recordNative(
        stage: String, asset: String, tensor: ImageNativeTensorObservation? = nil, memory: ImageMemoryObservation
    ) throws {
        guard let latest else { throw ImageExecutionError.invalidResource }
        let value = NativeCheckpoint(
            sequence: nativeSequence + 1, checkpointSequence: latest.sequence,
            processID: ProcessInfo.processInfo.processIdentifier, uptimeSeconds: ProcessInfo.processInfo.systemUptime,
            requestID: latest.report.requestID, profileID: latest.report.profileID,
            manifestSHA256: latest.report.manifestSHA256, component: latest.component, step: latest.step,
            asset: asset, stage: stage, tensor: tensor, memory: memory)
        try write(value, name: "native-checkpoint.json", replacing: nativeSequence > 0)
        nativeSequence += 1
    }

    private func write<Value: Encodable>(_ value: Value, name: String, replacing: Bool) throws {
        let values = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard directory.isFileURL, values.isDirectory == true, values.isSymbolicLink != true else {
            throw ImageExecutionError.invalidResource
        }
        let destination = directory.appending(path: name)
        if replacing {
            let existing = try destination.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard existing.isRegularFile == true, existing.isSymbolicLink != true else {
                throw ImageExecutionError.invalidResource
            }
        } else {
            guard !FileManager.default.fileExists(atPath: destination.path) else {
                throw ImageExecutionError.invalidResource
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: destination, options: .atomic)
    }
}
