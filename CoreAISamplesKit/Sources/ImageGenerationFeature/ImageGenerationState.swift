import Foundation
import ImageGenerationCore

public struct ImageGenerationTicket: Equatable, Hashable, Sendable {
    public let owner: UUID
    public let sequence: Int

    public init(owner: UUID, sequence: Int) {
        self.owner = owner
        self.sequence = sequence
    }
}

public enum ImageGenerationFailure: Equatable, Sendable {
    case unavailable, invalidImage, invalidInput, insufficientMemory, invalidModel, generationFailed, cleanupFailed,
        exportFailed
    case promptTooLong(Int)
}

public struct ImageReferencePreview: Equatable, Sendable, Identifiable {
    public let id: ImageGenerationTicket
    public let thumbnail: Data
    public let cropped: Data
    public let pixels: ImageGenerationPixels

    public init(id: ImageGenerationTicket, thumbnail: Data, cropped: Data, pixels: ImageGenerationPixels) {
        self.id = id
        self.thumbnail = thumbnail
        self.cropped = cropped
        self.pixels = pixels
    }
}

public struct ImageGeneratedAsset: Equatable, Sendable, Identifiable {
    public let id: UUID
    public let png: Data
    public let seed: UInt64
    public let report: ImageExecutionReport

    public init(id: UUID, png: Data, seed: UInt64, report: ImageExecutionReport) {
        self.id = id
        self.png = png
        self.seed = seed
        self.report = report
    }

    public var filename: String { "qwen-image-\(seed).png" }
}

public struct ImageExportTicket: Equatable, Sendable, Identifiable {
    public let id: UUID
    public let resultID: UUID

    public init(id: UUID, resultID: UUID) {
        self.id = id
        self.resultID = resultID
    }
}

public struct ImageGenerationState: Equatable, Sendable {
    public enum Availability: Equatable, Sendable { case needsModel, unsupported, ready }
    public enum Phase: Equatable, Sendable {
        case idle, readingReference, confirmingReference, generating, stopping, cleanupFailed
    }

    public let owner: UUID
    public let availability: Availability
    public let modelName: String
    public var presented = false
    public var phase: Phase = .idle
    public var prompt = ""
    public var seed = ""
    public var reference: ImageReferencePreview?
    public var proposedReference: ImageReferencePreview?
    public var result: ImageGeneratedAsset?
    public var progress: ImageExecutionPhase?
    public var failure: ImageGenerationFailure?
    public var export: ImageExportTicket?
    public var exportDismissed = false
    public var exportSucceeded = false
    public var sequence = 0
    public var operation: ImageGenerationTicket?
    public var requestID: UUID?

    public init(
        owner: UUID, availability: Availability, modelName: String = "Qwen Image 2.1 Turbo", retaining: Self? = nil
    ) {
        self.owner = owner
        self.availability = availability
        self.modelName = modelName
        prompt = retaining?.prompt ?? ""
        seed = retaining?.seed ?? ""
        reference = retaining?.reference
    }

    public var exportVisible: Bool { export != nil && !exportDismissed }
    public var editable: Bool { presented && phase == .idle && export == nil }
    public var seedIsValid: Bool {
        seed.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || UInt64(seed.trimmingCharacters(in: .whitespacesAndNewlines)).map { $0 <= Int64.max } == true
    }
    public var canGenerate: Bool {
        editable && availability == .ready && seedIsValid
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && prompt.utf8.count <= 262_144
    }
    public var canExport: Bool { editable && result != nil }
    public var canStop: Bool { presented && [.readingReference, .generating, .cleanupFailed].contains(phase) }
}
