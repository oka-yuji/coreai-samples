import Foundation

public enum ImageScalarType: String, Codable, Sendable {
    case float32, float16, bfloat16, int32, bool

    public var isFloatingPoint: Bool { self == .float32 || self == .float16 || self == .bfloat16 }
}

public enum ImageMetadataValue: Codable, Equatable, Sendable {
    case string(String)
    case number(Double)
    case boolean(Bool)
    case array([Self])
    case object([String: Self])
    case null

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .boolean(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([Self].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: Self].self))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .boolean(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

public struct ImageTensorContract: Codable, Equatable, Sendable {
    public let name: String
    public let shape: [Int]
    public let dtype: ImageScalarType
    public let layout: String

    public func elementCount() throws -> Int {
        guard !name.isEmpty, !shape.isEmpty, layout == "row-major" else {
            throw ImageGenerationError.invalidProfile("tensor descriptor")
        }
        var count = 1
        for dimension in shape {
            let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !overflow else { throw ImageGenerationError.invalidShape }
            count = next
        }
        return count
    }
}

public struct ImageGenerationProfile: Decodable, Sendable {
    public struct Source: Decodable, Sendable {
        public let model: String
        public let revision: String
        public let implementationRevision: String
        public let dependencyLockSha256: String
        public let converterSha256: [String: String]
    }

    public struct Workload: Decodable, Sendable {
        public let resolution: [Int]
        public let referenceResolution: [Int]
        public let batch: Int
        public let maximumReferences: Int
        public let processedTextCapacity: Int
        public let jointCapacity: Int
        public let operation: String
        public let cfg: Double
        public let sigmas: [Double]
        public let terminalSigma: Double
        public let schedulerComputeDtype: ImageGenerationPrecision
        public let schedulerOutputDtype: ImageGenerationPrecision
        public let schedulerDeltaDtype: String
        public let schedulerProductDtype: String
        public let modelTimestepRule: String
        public let castBack: String
        public let rounding: String
        public let prefixKvCache: Bool
        public let noiseShape: [Int]
        public let noiseDtype: ImageScalarType
        public let rngVersion: String
        public let denoiserFp16Clip: [Int]
        public let zeroCenteredNormDtype: ImageGenerationPrecision
    }

    public struct Component: Decodable, Sendable {
        public let asset: String
        public let function: String
        public let stateNames: [String]
        public let computeDtype: ImageGenerationPrecision
        public let weightEncoding: [String: ImageMetadataValue]
        public let inputs: [ImageTensorContract]
        public let outputs: [ImageTensorContract]
    }

    public static let revision = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"
    public static let componentNames: Set<String> = ["vision", "text", "denoiser", "vae_encoder", "vae_decoder"]
    public let format: Int
    public let id: String
    public let source: Source
    public let workload: Workload
    public let components: [String: Component]

    public static func decode(_ data: Data) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let profile = try decoder.decode(Self.self, from: data)
        try profile.validate()
        return profile
    }

    public func validate() throws {
        guard format == 2, !id.isEmpty, source.model == "Qwen/Qwen-Image-2.1-Turbo",
            source.revision == Self.revision,
            source.implementationRevision == "b800f828f3951c4d206e154411bcbfa693cd322e",
            imageDigestIsValid(source.dependencyLockSha256), !source.converterSha256.isEmpty,
            source.converterSha256.allSatisfy({ imageRelativePathIsValid($0.key) && imageDigestIsValid($0.value) })
        else { throw ImageGenerationError.invalidProfile("source identity") }
        let work = workload
        guard work.resolution == [512, 512], work.referenceResolution == [512, 512],
            work.batch == 1, work.maximumReferences == 1, work.processedTextCapacity == 512,
            work.jointCapacity == 2_304 || (work.jointCapacity == 1_536 && work.operation == "generate"),
            ["generate", "generate-and-edit"].contains(work.operation), work.cfg == 1,
            work.sigmas == [1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568],
            work.terminalSigma == 0, work.schedulerComputeDtype == .float32,
            work.schedulerDeltaDtype == "model-output", work.schedulerProductDtype == "model-output",
            work.modelTimestepRule == "sigma-f32-scale1000-cast-divide1000",
            work.castBack == "every-step-to-model-output", work.rounding == "nearest-ties-to-even",
            !work.prefixKvCache, work.noiseShape == [1, 1_024, 64], work.noiseDtype == .float32,
            !work.rngVersion.isEmpty, work.denoiserFp16Clip == [-65_504, 65_504], work.zeroCenteredNormDtype == .float32
        else { throw ImageGenerationError.invalidProfile("workload") }
        guard Set(components.keys) == Self.componentNames, Set(components.values.map(\.asset)).count == 5 else {
            throw ImageGenerationError.invalidProfile("five components")
        }
        for (kind, component) in components {
            guard imageRelativePathIsValid(component.asset), component.asset.hasSuffix(".aimodel"),
                component.function == "main", component.stateNames.isEmpty,
                case .string(let encoding)? = component.weightEncoding["kind"], !encoding.isEmpty
            else { throw ImageGenerationError.invalidProfile(kind) }
            try validate(component.inputs, expected: ImagePhysicalContract.inputs(kind, joint: work.jointCapacity))
            try validate(component.outputs, expected: ImagePhysicalContract.outputs(kind, joint: work.jointCapacity))
        }
    }

    private func validate(_ tensors: [ImageTensorContract], expected: [String: ImagePhysicalContract]) throws {
        guard Set(tensors.map(\.name)) == Set(expected.keys), tensors.count == expected.count else {
            throw ImageGenerationError.invalidProfile("tensor names")
        }
        for tensor in tensors {
            _ = try tensor.elementCount()
            guard let specification = expected[tensor.name], tensor.shape == specification.shape,
                specification.dtype.isFloatingPoint ? tensor.dtype.isFloatingPoint : tensor.dtype == specification.dtype
            else { throw ImageGenerationError.invalidProfile(tensor.name) }
        }
    }
}

struct ImagePhysicalContract {
    let shape: [Int]
    let dtype: ImageScalarType

    init(_ shape: [Int], _ dtype: ImageScalarType = .float32) {
        self.shape = shape
        self.dtype = dtype
    }

    static func inputs(_ kind: String, joint: Int) -> [String: Self] {
        switch kind {
        case "vision": ["pixel_values": Self([1_024, 1_536])]
        case "text":
            [
                "input_ids": Self([1, 512], .int32), "vision": Self([1, 512, 4_096]),
                "deepstack": Self([3, 1, 512, 4_096]), "cosine": Self([1, 512, 128]), "sine": Self([1, 512, 128]),
                "mask": Self([1, 1, 512, 512], .bool), "visual_mask": Self([1, 512], .bool),
            ]
        case "denoiser":
            [
                "latents": Self([1, joint == 2_304 ? 2_048 : 1_024, 64]), "text": Self([1, 512, 4_096]),
                "timestep": Self([1]), "text_index": Self([joint], .int32), "image_index": Self([joint], .int32),
                "image_mask": Self([joint], .bool), "cosine": Self([1, joint, 1, 64]), "sine": Self([1, joint, 1, 64]),
                "mask": Self([1, 1, joint, joint], .bool), "target": Self([joint], .bool),
            ]
        case "vae_encoder": ["value": Self([1, 4, 1, 512, 512])]
        default: ["value": Self([1, 64, 1, 32, 32])]
        }
    }

    static func outputs(_ kind: String, joint: Int) -> [String: Self] {
        switch kind {
        case "vision":
            [
                "merged": Self([256, 4_096]), "deepstack_0": Self([256, 4_096]),
                "deepstack_1": Self([256, 4_096]), "deepstack_2": Self([256, 4_096]),
            ]
        case "text": ["hidden": Self([1, 512, 4_096])]
        case "denoiser": ["noise": Self([1, joint, 64])]
        case "vae_encoder": ["result": Self([1, 64, 1, 32, 32])]
        default: ["result": Self([1, 4, 1, 512, 512])]
        }
    }
}

func imageRelativePathIsValid(_ path: String) -> Bool {
    !path.isEmpty && !path.hasPrefix("/") && !path.contains("\\")
        && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy {
            !$0.isEmpty && $0 != "." && $0 != ".."
        }
}

func imageDigestIsValid(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}
