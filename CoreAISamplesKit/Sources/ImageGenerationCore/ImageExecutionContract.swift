import Foundation

public enum ImageModelComponent: String, CaseIterable, Codable, Sendable {
    case vision, text, denoiser
    case vaeEncoder = "vae_encoder"
    case vaeDecoder = "vae_decoder"
}

public enum ImageExecutionError: Error, Equatable, Sendable {
    case busy
    case released
    case permissionDenied
    case invalidResource
    case editingUnsupported
    case invalidTensor(String)
    case memoryUnavailable
    case budgetNotConfigured
    case insufficientMemory(component: ImageModelComponent, required: UInt64, available: UInt64)
    case releaseUnconfirmed(String)
}

public enum ImageTensorData: Equatable, Sendable {
    case floats([Float])
    case integers([Int32])
    case mask([UInt8])

    public var count: Int {
        switch self {
        case .floats(let values): values.count
        case .integers(let values): values.count
        case .mask(let values): values.count
        }
    }

    public func validate(_ contract: ImageTensorContract) throws {
        guard count == (try contract.elementCount()) else { throw ImageExecutionError.invalidTensor(contract.name) }
        let valid: Bool
        switch self {
        case .floats(let values): valid = contract.dtype == .float32 && values.allSatisfy({ $0.isFinite })
        case .integers: valid = contract.dtype == .int32
        case .mask(let values): valid = contract.dtype == .bool && Self.isBooleanMask(values)
        }
        guard valid else { throw ImageExecutionError.invalidTensor(contract.name) }
    }

    public func floats() throws -> [Float] {
        guard case .floats(let values) = self else { throw ImageExecutionError.invalidTensor("floating output") }
        return values
    }

    private static func isBooleanMask(_ values: [UInt8]) -> Bool {
        values.withUnsafeBytes { bytes in
            var offset = 0
            while bytes.count - offset >= 8 {
                if bytes.loadUnaligned(fromByteOffset: offset, as: UInt64.self) & 0xFEFE_FEFE_FEFE_FEFE != 0 {
                    return false
                }
                offset += 8
            }
            while offset < bytes.count {
                if bytes[offset] > 1 { return false }
                offset += 1
            }
            return true
        }
    }
}

public enum ImageMemoryScope: String, Codable, Sendable {
    case processDirtyBudget
    case estimatedSystemHeadroom
    case unavailable
}

public struct ImageMemoryObservation: Codable, Equatable, Sendable {
    public let scope: ImageMemoryScope
    public let availableBytes: UInt64?
    public let footprintBytes: UInt64?
    public let residentBytes: UInt64?
    public let kernelLifetimePeakBytes: UInt64?
    public let visibleMetalBytes: UInt64?
    public let physicalMemoryBytes: UInt64
    public let thermalState: Int
    public let lowPowerMode: Bool

    public init(
        scope: ImageMemoryScope, availableBytes: UInt64?, footprintBytes: UInt64?, residentBytes: UInt64? = nil,
        kernelLifetimePeakBytes: UInt64? = nil, visibleMetalBytes: UInt64? = nil,
        physicalMemoryBytes: UInt64, thermalState: Int = 0, lowPowerMode: Bool = false
    ) {
        self.scope = scope
        self.availableBytes = availableBytes
        self.footprintBytes = footprintBytes
        self.residentBytes = residentBytes
        self.kernelLifetimePeakBytes = kernelLifetimePeakBytes
        self.visibleMetalBytes = visibleMetalBytes
        self.physicalMemoryBytes = physicalMemoryBytes
        self.thermalState = thermalState
        self.lowPowerMode = lowPowerMode
    }

    public func budgetBytes() throws -> UInt64 {
        guard let availableBytes, let footprintBytes, physicalMemoryBytes > 0 else {
            throw ImageExecutionError.memoryUnavailable
        }
        switch scope {
        case .processDirtyBudget:
            let (total, overflow) = availableBytes.addingReportingOverflow(footprintBytes)
            guard !overflow, total > 0 else { throw ImageExecutionError.memoryUnavailable }
            return total
        case .estimatedSystemHeadroom: return physicalMemoryBytes
        case .unavailable: throw ImageExecutionError.memoryUnavailable
        }
    }
}

public enum ImageMemoryPhase: String, Codable, Sendable {
    case loading, inference
}

public struct ImageComponentMemoryBudget: Codable, Equatable, Sendable {
    public let loadingAdditionalBytes: UInt64
    public let inferenceAdditionalBytes: UInt64

    public init(loadingAdditionalBytes: UInt64, inferenceAdditionalBytes: UInt64) throws {
        guard loadingAdditionalBytes > 0, inferenceAdditionalBytes > 0 else {
            throw ImageExecutionError.budgetNotConfigured
        }
        self.loadingAdditionalBytes = loadingAdditionalBytes
        self.inferenceAdditionalBytes = inferenceAdditionalBytes
    }
}

public struct ImageMemoryBudget: Codable, Equatable, Sendable {
    public let profileID: String
    public let manifestSHA256: String
    public let components: [ImageModelComponent: ImageComponentMemoryBudget]

    public init(
        profileID: String, manifestSHA256: String, components: [ImageModelComponent: ImageComponentMemoryBudget]
    ) throws {
        guard !profileID.isEmpty, imageDigestIsValid(manifestSHA256),
            Set(components.keys) == Set(ImageModelComponent.allCases),
            components.values.allSatisfy({ $0.loadingAdditionalBytes > 0 && $0.inferenceAdditionalBytes > 0 })
        else { throw ImageExecutionError.budgetNotConfigured }
        self.profileID = profileID
        self.manifestSHA256 = manifestSHA256
        self.components = components
    }

    private enum CodingKeys: String, CodingKey {
        case profileID, manifestSHA256, components
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decoded = try container.decode([String: ImageComponentMemoryBudget].self, forKey: .components)
        var components: [ImageModelComponent: ImageComponentMemoryBudget] = [:]
        for (name, budget) in decoded {
            guard let kind = ImageModelComponent(rawValue: name) else { throw ImageExecutionError.budgetNotConfigured }
            components[kind] = budget
        }
        try self.init(
            profileID: container.decode(String.self, forKey: .profileID),
            manifestSHA256: container.decode(String.self, forKey: .manifestSHA256), components: components)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(profileID, forKey: .profileID)
        try container.encode(manifestSHA256, forKey: .manifestSHA256)
        try container.encode(
            Dictionary(uniqueKeysWithValues: components.map { ($0.key.rawValue, $0.value) }), forKey: .components)
    }

    public func check(
        _ component: ImageModelComponent, phase: ImageMemoryPhase, observation: ImageMemoryObservation
    ) throws -> UInt64 {
        guard let componentBudget = components[component],
            componentBudget.loadingAdditionalBytes > 0, componentBudget.inferenceAdditionalBytes > 0
        else { throw ImageExecutionError.budgetNotConfigured }
        let additional =
            phase == .loading
            ? componentBudget.loadingAdditionalBytes : componentBudget.inferenceAdditionalBytes
        let budget = try observation.budgetBytes()
        let reserve = max(536_870_912, budget / 5 + (budget.isMultiple(of: 5) ? 0 : 1))
        let (required, overflow) = additional.addingReportingOverflow(reserve)
        guard !overflow else { throw ImageExecutionError.budgetNotConfigured }
        guard let available = observation.availableBytes else { throw ImageExecutionError.memoryUnavailable }
        guard available >= required else {
            throw ImageExecutionError.insufficientMemory(component: component, required: required, available: available)
        }
        return reserve
    }
}

public struct ImageGenerationRequest: Sendable {
    public let id: UUID
    public let prompt: String
    public let seed: UInt64
    public let reference: ImageGenerationPixels?
    public let initialNoise: [Float]?

    public init(
        id: UUID, prompt: String, seed: UInt64, reference: ImageGenerationPixels? = nil, initialNoise: [Float]? = nil
    ) {
        self.id = id
        self.prompt = prompt
        self.seed = seed
        self.reference = reference
        self.initialNoise = initialNoise
    }

    public func validate() throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ImageGenerationError.invalidTokens
        }
        guard prompt.utf8.count <= 262_144 else { throw ImageGenerationError.inputTooLarge(prompt.utf8.count) }
        guard seed <= Int64.max else { throw ImageGenerationError.invalidSeed }
        if let reference, reference.width != 512 || reference.height != 512 {
            throw ImageGenerationError.invalidShape
        }
        if let initialNoise {
            guard initialNoise.count == ImageEuler.latentCount else { throw ImageGenerationError.invalidShape }
            guard initialNoise.allSatisfy({ $0.isFinite }) else { throw ImageGenerationError.nonFinite }
        }
    }
}

public enum ImageExecutionPhase: Equatable, Sendable {
    case preparing
    case loading(ImageModelComponent)
    case running(ImageModelComponent, step: Int?)
    case releasing(ImageModelComponent)
}

public struct ImageExecutionProgress: Equatable, Sendable {
    public let requestID: UUID
    public let phase: ImageExecutionPhase

    public init(requestID: UUID, phase: ImageExecutionPhase) {
        self.requestID = requestID
        self.phase = phase
    }
}
