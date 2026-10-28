import CoreAI
import Foundation
import ImageGenerationCore

enum ImageModelLoading: String, Codable, Sendable {
    case explicitCache, automatic
}

protocol ImageComponentBackend: Actor {
    func load(directory: URL, component: ImageGenerationProfile.Component) async throws -> ImageComponentLoadMetrics
    func call(_ values: [String: ImageTensorData]) async throws -> [String: ImageTensorData]
    func unload() async throws
}

actor CoreAIImageBackend: ImageComponentBackend {
    private struct Loaded {
        let model: AIModel
        let function: InferenceFunction
        let component: ImageGenerationProfile.Component
    }

    private var loaded: Loaded?
    private var busy = false
    private let journal: ImageExecutionJournal?
    private let loadingMode: ImageModelLoading

    init(journal: ImageExecutionJournal? = nil, loadingMode: ImageModelLoading = .explicitCache) {
        self.journal = journal
        self.loadingMode = loadingMode
    }

    func load(directory: URL, component: ImageGenerationProfile.Component) async throws -> ImageComponentLoadMetrics {
        guard !busy, loaded == nil else { throw ImageExecutionError.busy }
        busy = true
        defer { busy = false }
        let url = directory.appending(path: component.asset)
        let options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        let model: AIModel
        let cacheHit: Bool?
        let lookupSeconds: Double
        let specializationSeconds: Double?
        let initializationSeconds: Double?
        switch loadingMode {
        case .automatic:
            let start = ContinuousClock.now
            model = try await AIModel(contentsOf: url, options: options)
            initializationSeconds = elapsed(since: start)
            cacheHit = nil
            lookupSeconds = 0
            specializationSeconds = nil
        case .explicitCache:
            initializationSeconds = nil
            let lookupStart = ContinuousClock.now
            let cached = try AIModelCache.default.model(for: url, options: options)
            lookupSeconds = elapsed(since: lookupStart)
            cacheHit = cached != nil
            if let cached {
                model = cached
                specializationSeconds = nil
            } else {
                let start = ContinuousClock.now
                model = try await AIModel.specialize(contentsOf: url, options: options)
                specializationSeconds = elapsed(since: start)
            }
        }
        guard let descriptor = model.functionDescriptor(for: component.function), descriptor.stateNames.isEmpty,
            Set(descriptor.inputNames) == Set(component.inputs.map(\.name)),
            Set(descriptor.outputNames) == Set(component.outputs.map(\.name))
        else { throw ImageExecutionError.invalidResource }
        for contract in component.inputs {
            try ImageCoreAIBuffers.check(
                descriptor.inputDescriptor(of: contract.name), contract: contract, writable: true)
        }
        for contract in component.outputs {
            try ImageCoreAIBuffers.check(descriptor.outputDescriptor(of: contract.name), contract: contract)
        }
        try await journal?.recordNative(
            stage: "function-load-before", asset: component.asset, memory: ImageMemorySampler.capture())
        let loadStart = ContinuousClock.now
        guard let function = try model.loadFunction(named: component.function) else {
            throw ImageExecutionError.invalidResource
        }
        loaded = Loaded(model: model, function: function, component: component)
        let loadSeconds = elapsed(since: loadStart)
        try await journal?.recordNative(
            stage: "function-load-after", asset: component.asset, memory: ImageMemorySampler.capture())
        return ImageComponentLoadMetrics(
            cacheHit: cacheHit, cacheLookupSeconds: lookupSeconds, specializationSeconds: specializationSeconds,
            functionLoadSeconds: loadSeconds, initializationSeconds: initializationSeconds,
            loadingMode: loadingMode.rawValue)
    }

    func call(_ values: [String: ImageTensorData]) async throws -> [String: ImageTensorData] {
        guard !busy else { throw ImageExecutionError.busy }
        guard let loaded else { throw ImageExecutionError.invalidResource }
        busy = true
        defer { busy = false }
        return try await Self.evaluate(loaded, values: values, journal: journal)
    }

    func unload() throws {
        guard !busy else { throw ImageExecutionError.busy }
        loaded = nil
    }

    private static func evaluate(
        _ loaded: Loaded, values: [String: ImageTensorData], journal: ImageExecutionJournal?
    ) async throws -> [String: ImageTensorData] {
        guard Set(values.keys) == Set(loaded.component.inputs.map(\.name)) else {
            throw ImageExecutionError.invalidResource
        }
        var inputs: [String: NDArray] = [:]
        for contract in loaded.component.inputs {
            guard let value = values[contract.name],
                case .ndArray(let descriptor)? = loaded.function.descriptor.inputDescriptor(of: contract.name)
            else { throw ImageExecutionError.invalidTensor(contract.name) }
            let observation = ImageNativeTensorObservation(
                name: contract.name, shape: descriptor.shape, strides: descriptor.preferredStrides,
                minimumByteCount: descriptor.minimumByteCount)
            try await journal?.recordNative(
                stage: "input-before", asset: loaded.component.asset, tensor: observation,
                memory: ImageMemorySampler.capture())
            inputs[contract.name] = try ImageCoreAIBuffers.make(
                value, contract: contract, strides: descriptor.preferredStrides)
            try await journal?.recordNative(
                stage: "input-after", asset: loaded.component.asset, tensor: observation,
                memory: ImageMemorySampler.capture())
        }
        try await journal?.recordNative(
            stage: "run-before", asset: loaded.component.asset, memory: ImageMemorySampler.capture())
        var outputs = try await loaded.function.run(inputs: inputs)
        try await journal?.recordNative(
            stage: "run-after", asset: loaded.component.asset, memory: ImageMemorySampler.capture())
        var result: [String: ImageTensorData] = [:]
        for contract in loaded.component.outputs {
            guard let array = outputs.remove(contract.name)?.ndArray else {
                throw ImageExecutionError.invalidTensor(contract.name)
            }
            result[contract.name] = .floats(try ImageCoreAIBuffers.floats(array, contract: contract))
        }
        return result
    }

    private func elapsed(since start: ContinuousClock.Instant) -> Double {
        (ContinuousClock.now - start) / .seconds(1)
    }
}

enum ImageCoreAIBuffers {
    static func scalar(_ type: ImageScalarType) throws -> NDArray.ScalarType {
        switch type {
        case .float32: .float32
        case .int32: .int32
        case .bool: .bool
        case .float16, .bfloat16: throw ImageExecutionError.invalidTensor("unsupported physical precision")
        }
    }

    static func check(_ value: InferenceValue.Descriptor?, contract: ImageTensorContract, writable: Bool = false) throws
    {
        guard case .ndArray(let descriptor)? = value, !descriptor.hasDynamicShape,
            descriptor.shape == contract.shape, descriptor.scalarType == (try scalar(contract.dtype)),
            descriptor.interleaveLayout == nil
        else { throw ImageExecutionError.invalidTensor(contract.name) }
        try validateStrides(descriptor.preferredStrides, shape: contract.shape, writable: writable)
    }

    static func make(_ value: ImageTensorData, contract: ImageTensorContract, strides: [Int]) throws -> NDArray {
        try value.validate(contract)
        try validateStrides(strides, shape: contract.shape, writable: true)
        var array = NDArray(shape: contract.shape, scalarType: try scalar(contract.dtype), strides: strides)
        switch value {
        case .floats(let values):
            try values.withUnsafeBufferPointer { try fill(&array, bytes: UnsafeRawBufferPointer($0), size: 4) }
        case .integers(let values):
            try values.withUnsafeBufferPointer { try fill(&array, bytes: UnsafeRawBufferPointer($0), size: 4) }
        case .mask(let values):
            try values.withUnsafeBufferPointer { try fill(&array, bytes: UnsafeRawBufferPointer($0), size: 1) }
        }
        return array
    }

    static func floats(_ array: NDArray, contract: ImageTensorContract) throws -> [Float] {
        guard array.scalarType == .float32, contract.dtype == .float32, array.shape == contract.shape,
            array.interleaveLayout == nil
        else { throw ImageExecutionError.invalidTensor(contract.name) }
        let count = try contract.elementCount()
        let shape = array.shape
        let strides = array.strides
        try validateStrides(strides, shape: shape)
        return array.rawView().withUnsafeBytes { pointer, _, _ in
            if contiguous(shape: shape, strides: strides) {
                return Array(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Float.self), count: count))
            }
            return (0..<count).map {
                pointer.load(fromByteOffset: offset($0, shape: shape, strides: strides) * 4, as: Float.self)
            }
        }
    }

    private static func fill(_ array: inout NDArray, bytes: UnsafeRawBufferPointer, size: Int) throws {
        guard let source = bytes.baseAddress, !bytes.isEmpty, bytes.count.isMultiple(of: size) else {
            throw ImageGenerationError.invalidShape
        }
        let shape = array.shape
        let strides = array.strides
        array.mutableRawView().withUnsafeMutableBytes { destination, _, _ in
            if contiguous(shape: shape, strides: strides) {
                destination.copyMemory(from: source, byteCount: bytes.count)
            } else {
                for index in 0..<(bytes.count / size) {
                    destination.advanced(by: offset(index, shape: shape, strides: strides) * size)
                        .copyMemory(from: source.advanced(by: index * size), byteCount: size)
                }
            }
        }
    }

    private static func validateStrides(_ strides: [Int], shape: [Int], writable: Bool = false) throws {
        guard strides.count == shape.count, !shape.isEmpty, shape.allSatisfy({ $0 > 0 }),
            strides.allSatisfy({ $0 >= 0 })
        else { throw ImageGenerationError.invalidShape }
        var maximum = 0
        for (dimension, stride) in zip(shape, strides) {
            let (term, multiplied) = (dimension - 1).multipliedReportingOverflow(by: stride)
            let (sum, added) = maximum.addingReportingOverflow(term)
            guard !multiplied, !added, sum < Int.max / 4 else { throw ImageGenerationError.invalidShape }
            maximum = sum
        }
        if writable {
            var occupied = 1
            for (dimension, stride) in zip(shape, strides).filter({ $0.0 > 1 }).sorted(by: { $0.1 < $1.1 }) {
                guard stride >= occupied else { throw ImageGenerationError.invalidShape }
                occupied += (dimension - 1) * stride
            }
        }
    }

    private static func contiguous(shape: [Int], strides: [Int]) -> Bool {
        var stride = 1
        for index in shape.indices.reversed() {
            if shape[index] > 1, strides[index] != stride { return false }
            stride *= shape[index]
        }
        return true
    }

    private static func offset(_ index: Int, shape: [Int], strides: [Int]) -> Int {
        var remainder = index
        var result = 0
        for axis in shape.indices.reversed() {
            result += (remainder % shape[axis]) * strides[axis]
            remainder /= shape[axis]
        }
        return result
    }
}
