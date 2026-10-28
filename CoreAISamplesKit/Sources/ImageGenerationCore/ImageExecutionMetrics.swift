import Foundation

public struct ImageComponentLoadMetrics: Codable, Equatable, Sendable {
    public let cacheHit: Bool?
    public let cacheLookupSeconds: Double
    public let specializationSeconds: Double?
    public let functionLoadSeconds: Double
    public let initializationSeconds: Double?
    public let loadingMode: String?

    public init(
        cacheHit: Bool?, cacheLookupSeconds: Double, specializationSeconds: Double?, functionLoadSeconds: Double,
        initializationSeconds: Double? = nil, loadingMode: String? = nil
    ) {
        self.cacheHit = cacheHit
        self.cacheLookupSeconds = cacheLookupSeconds
        self.specializationSeconds = specializationSeconds
        self.functionLoadSeconds = functionLoadSeconds
        self.initializationSeconds = initializationSeconds
        self.loadingMode = loadingMode
    }
}

public enum ImageMeasuredOperation: String, Codable, Sendable {
    case loading, inference, release
}

public struct ImagePhaseMetric: Codable, Equatable, Sendable {
    public let component: ImageModelComponent
    public let operation: ImageMeasuredOperation
    public let step: Int?
    public let seconds: Double
    public let before: ImageMemoryObservation
    public let after: ImageMemoryObservation
    public let reserveBytes: UInt64?
    public let additionalBudgetBytes: UInt64?
    public let load: ImageComponentLoadMetrics?
    public let error: String?

    public init(
        component: ImageModelComponent, operation: ImageMeasuredOperation, step: Int? = nil, seconds: Double,
        before: ImageMemoryObservation, after: ImageMemoryObservation, reserveBytes: UInt64? = nil,
        additionalBudgetBytes: UInt64? = nil, load: ImageComponentLoadMetrics? = nil, error: String? = nil
    ) {
        self.component = component
        self.operation = operation
        self.step = step
        self.seconds = seconds
        self.before = before
        self.after = after
        self.reserveBytes = reserveBytes
        self.additionalBudgetBytes = additionalBudgetBytes
        self.load = load
        self.error = error
    }
}

public struct ImageExecutionReport: Codable, Equatable, Sendable {
    public enum Status: String, Codable, Sendable {
        case running, completed, cancelled, failed, releaseUnconfirmed
    }

    public let requestID: UUID
    public let profileID: String
    public let manifestSHA256: String
    public let noiseSource: String
    public let diagnosticsEnabled: Bool
    public var status: Status
    public var totalSeconds: Double
    public var phases: [ImagePhaseMetric]
    public var error: String?

    public init(
        requestID: UUID, profileID: String, manifestSHA256: String, noiseSource: String, diagnosticsEnabled: Bool
    ) {
        self.requestID = requestID
        self.profileID = profileID
        self.manifestSHA256 = manifestSHA256
        self.noiseSource = noiseSource
        self.diagnosticsEnabled = diagnosticsEnabled
        status = .running
        totalSeconds = 0
        phases = []
    }
}

public struct ImageGenerationResult: Equatable, Sendable {
    public let requestID: UUID
    public let pixels: ImageGenerationPixels
    public let report: ImageExecutionReport

    public init(requestID: UUID, pixels: ImageGenerationPixels, report: ImageExecutionReport) {
        self.requestID = requestID
        self.pixels = pixels
        self.report = report
    }
}
