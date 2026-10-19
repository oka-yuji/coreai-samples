import Accelerate
import CoreAI
import Foundation

public enum EngineError: Error, Equatable {
    case missingFunction(String)
    case missingInput(String)
    case missingOutput
    case unsupportedState(String)
    case contextOverflow(needed: Int, limit: Int)
    case checkpointMismatch
    case unloaded
}

public final class Engine {
    public struct LoadStats: Sendable {
        public let cached: Bool
        public let specializeSeconds: Double
        public let totalSeconds: Double
        public let requestedComputeUnit: String?
        public let expectsFrequentReshapes: Bool?
    }

    public struct Stats: Sendable {
        public var prefillTokens = 0
        public var prefillSeconds = 0.0
        public var decodeTokens = 0
        public var decodeSeconds = 0.0
        public var prepareSeconds = 0.0

        public init() {}
    }

    public struct Checkpoint: Sendable {
        public let tokens: [Int32]
        let fixedStates: [Data]
        var logits: [Float16] = []

        public var stateBytes: Int { fixedStates.reduce(0) { $0 + $1.count } }
    }

    static let prefillChunkSize = ProcessInfo.processInfo.physicalMemory >= 64 << 30 ? 4096 : 2048

    public let vocabSize: Int
    public let contextLimit: Int
    public let hasPrefillFunction: Bool
    public let isHybrid: Bool
    public let loadStats: LoadStats
    public var stats = Stats()

    private var main: InferenceFunction?
    private var prefillFunction: InferenceFunction?
    private var context: Context
    private let logitsName: String
    private let logitsDescriptor: NDArrayDescriptor
    private var logitsArray: NDArray
    private var logitsRows = 1
    private var logitsOffset = 0
    private(set) var hasLogits = false
    private var scratch: [Float]

    public var tokens: [Int32] { context.tokens }
    public var kvCapacity: Int { context.capacity }

    public init(bundle: ModelBundle, contextLimit: Int? = nil) async throws {
        let clock = ContinuousClock()
        let start = clock.now
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        var specializeSeconds = 0.0
        let model: AIModel
        if let cached = (try? AIModelCache.default.model(for: bundle.modelURL, options: options)) ?? nil {
            model = cached
        } else {
            if let width = Self.mainOutputWidth(bundle.modelURL), width != bundle.vocabSize {
                throw BundleError.notLanguageModel(outputWidth: width, vocabSize: bundle.vocabSize)
            }
            let specializeStart = clock.now
            model = try await AIModel.specialize(contentsOf: bundle.modelURL, options: options)
            specializeSeconds = (clock.now - specializeStart) / .seconds(1)
        }

        guard let descriptor = model.functionDescriptor(for: "main"),
            let main = try model.loadFunction(named: "main")
        else { throw EngineError.missingFunction("main") }
        for name in ["input_ids", "position_ids"] where !descriptor.inputNames.contains(name) {
            throw EngineError.missingInput(name)
        }
        guard let logitsName = descriptor.outputNames.first,
            case .ndArray(let logitsDescriptor) = descriptor.outputDescriptor(of: logitsName),
            logitsDescriptor.scalarType == .float16
        else { throw EngineError.missingOutput }
        if let width = logitsDescriptor.shape.last, width >= 0, width != bundle.vocabSize {
            throw BundleError.notLanguageModel(outputWidth: width, vocabSize: bundle.vocabSize)
        }
        let prefillFunction = model.functionNames.contains("prefill") ? try model.loadFunction(named: "prefill") : nil

        let limit = min(contextLimit ?? bundle.maxContextLength, bundle.maxContextLength)
        var buffers: [StateBuffer] = []
        for name in descriptor.stateNames {
            guard case .ndArray(let state) = descriptor.stateDescriptor(of: name) else {
                throw EngineError.unsupportedState(name)
            }
            buffers.append(try StateBuffer(name: name, descriptor: state, capacity: min(256, limit)))
        }

        self.main = main
        self.prefillFunction = prefillFunction
        self.logitsName = logitsName
        self.logitsDescriptor = logitsDescriptor
        self.logitsArray = NDArray(descriptor: logitsDescriptor.resolvingDynamicDimensions([1, 1, bundle.vocabSize]))
        self.context = Context(buffers: buffers, capacity: min(256, limit), limit: limit)
        self.vocabSize = bundle.vocabSize
        self.contextLimit = limit
        self.hasPrefillFunction = prefillFunction != nil
        self.isHybrid = buffers.contains { $0.sequenceDim == nil }
        self.scratch = [Float](repeating: 0, count: bundle.vocabSize)
        self.loadStats = LoadStats(
            cached: specializeSeconds == 0,
            specializeSeconds: specializeSeconds,
            totalSeconds: (clock.now - start) / .seconds(1),
            requestedComputeUnit: Self.computeUnitName(options.preferredComputeUnitKind),
            expectsFrequentReshapes: options.expectFrequentReshapes)
    }

    static func computeUnitName(_ kind: ComputeUnitKind?) -> String {
        guard let kind else { return "default" }
        switch kind {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "neuralEngine"
        @unknown default: return "other"
        }
    }

    static func mainOutputWidth(_ modelURL: URL) -> Int? {
        let summary = try? AIModelAsset(contentsOf: modelURL).summary(includingStatistics: false)
        return (summary?.functions.first { $0.name == "main" }?.outputs.first?.typeName).flatMap(lastDimension)
    }

    static func lastDimension(_ typeName: String) -> Int? {
        guard typeName.hasSuffix(")"), let shape = typeName.dropLast().split(separator: ",").last else { return nil }
        return shape.split(separator: "\u{D7}").last.flatMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    public static func deleteSpecializationCache(bundle: ModelBundle) throws {
        try AIModelCache.default.deleteEntries(for: bundle.modelURL)
    }

    public func prefill(_ input: [Int32], usePrefillFunction: Bool = true) async throws {
        try await prefill(input, usePrefillFunction: usePrefillFunction, onProgress: { _ in })
    }

    public func prefill(
        _ input: [Int32], usePrefillFunction: Bool, onProgress: @Sendable (Int) -> Void
    ) async throws {
        precondition(!input.isEmpty)
        guard let main else { throw EngineError.unloaded }
        try context.checkRoom(input.count)
        let start = ContinuousClock.now
        var rest = input[...]
        onProgress(0)
        if let prefillFunction, usePrefillFunction {
            while rest.count > 1 {
                let chunk = rest.prefix(min(Self.prefillChunkSize, rest.count - 1))
                try await forward(prefillFunction, chunk, logits: false)
                rest = rest.dropFirst(chunk.count)
                onProgress(input.count - rest.count)
            }
        }
        while !rest.isEmpty {
            let chunk = rest.prefix(Self.prefillChunkSize)
            try await forward(main, chunk, logits: true)
            rest = rest.dropFirst(chunk.count)
            onProgress(input.count - rest.count)
        }
        stats.prefillTokens += input.count
        stats.prefillSeconds += (ContinuousClock.now - start) / .seconds(1)
    }

    public func step(_ token: Int32) async throws {
        guard let main else { throw EngineError.unloaded }
        try context.checkRoom(1)
        let start = ContinuousClock.now
        try await forward(main, [token][...], logits: true)
        stats.decodeTokens += 1
        stats.decodeSeconds += (ContinuousClock.now - start) / .seconds(1)
    }

    public func generate(
        _ input: [Int32],
        maxTokens: Int,
        stopTokens: Set<Int32>,
        onToken: (Int32) -> Void = { _ in }
    ) async throws -> [Int32] {
        try await prefill(input)
        var output: [Int32] = []
        while true {
            let token = greedyToken()
            output.append(token)
            onToken(token)
            if output.count >= maxTokens || stopTokens.contains(token) { return output }
            try await step(token)
        }
    }

    public func greedyToken() -> Int32 {
        withLogits { Self.argmax($0, scratch: &scratch) }
    }

    public func logits() -> [Float16] {
        withLogits { Array($0) }
    }

    public func checkpoint() -> Checkpoint {
        var point = context.checkpoint()
        if hasLogits { point.logits = logits() }
        return point
    }

    public func restore(_ checkpoint: Checkpoint) throws {
        try context.restore(checkpoint)
        hasLogits = checkpoint.logits.count == vocabSize
        guard hasLogits else { return }
        var array = NDArray(descriptor: logitsDescriptor.resolvingDynamicDimensions([1, 1, vocabSize]))
        array.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, _ in
            checkpoint.logits.withUnsafeBufferPointer { pointer.update(from: $0.baseAddress!, count: $0.count) }
        }
        logitsArray = array
        logitsRows = 1
        logitsOffset = 0
    }

    public func reset() {
        context.reset()
        hasLogits = false
    }

    public func unload() {
        main = nil
        prefillFunction = nil
        context.buffers = []
        context.tokens = []
        logitsArray = NDArray(shape: [1], scalarType: .float16)
        hasLogits = false
    }

    func withLogits<R>(_ body: (UnsafeBufferPointer<Float16>) -> R) -> R {
        precondition(hasLogits, "no logits for the current position")
        let array = logitsArray
        let offset = logitsOffset
        return array.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
            body(UnsafeBufferPointer(start: pointer + offset, count: vocabSize))
        }
    }

    private func forward(_ function: InferenceFunction, _ chunk: ArraySlice<Int32>, logits: Bool) async throws {
        let prepareStart = ContinuousClock.now
        let end = context.tokens.count + chunk.count
        context.reserve(end)
        let inputs = [
            "input_ids": Self.row(chunk.count) { chunk[chunk.startIndex + $0] },
            "position_ids": Self.row(end) { Int32($0) },
        ]
        var stateViews = InferenceFunction.MutableViews()
        for index in context.buffers.indices {
            let name = context.buffers[index].name
            stateViews.insert(
                unsafe _overrideLifetime(context.buffers[index].array.mutableRawView(), borrowing: self), for: name)
        }
        var outputViews = InferenceFunction.MutableViews()
        if logits {
            if logitsRows != chunk.count {
                logitsArray = NDArray(
                    descriptor: logitsDescriptor.resolvingDynamicDimensions([1, chunk.count, vocabSize]))
                logitsRows = chunk.count
            }
            outputViews.insert(unsafe _overrideLifetime(logitsArray.mutableRawView(), borrowing: self), for: logitsName)
            logitsOffset = (chunk.count - 1) * logitsArray.strides[1]
        }
        stats.prepareSeconds += (ContinuousClock.now - prepareStart) / .seconds(1)
        _ = try await function.run(inputs: inputs, states: stateViews, outputViews: outputViews)
        context.tokens.append(contentsOf: chunk)
        hasLogits = logits
    }

    static func row(_ count: Int, _ value: (Int) -> Int32) -> NDArray {
        var array = NDArray(shape: [1, count], scalarType: .int32)
        array.mutableView(as: Int32.self).withUnsafeMutablePointer { pointer, _, _ in
            for i in 0..<count { pointer[i] = value(i) }
        }
        return array
    }

    static func argmax(_ values: UnsafeBufferPointer<Float16>, scratch: inout [Float]) -> Int32 {
        precondition(scratch.count >= values.count)
        var index: vDSP_Length = 0
        var maximum: Float = 0
        scratch.withUnsafeMutableBufferPointer { floats in
            for i in values.indices { floats[i] = Float(values[i]) }
            vDSP_maxvi(floats.baseAddress!, 1, &maximum, &index, vDSP_Length(values.count))
        }
        return Int32(index)
    }

    static func grownCapacity(current: Int, needed: Int, limit: Int) -> Int {
        var capacity = max(current, 1)
        while capacity < needed { capacity = min(capacity * 2, limit) }
        return capacity
    }
}

struct StateBuffer {
    let name: String
    let scalarType: NDArray.ScalarType
    let sequenceDim: Int?
    let layout: @Sendable ([Int]) -> (strides: [Int], byteCount: Int)
    var shape: [Int]
    var strides: [Int]
    var byteCount: Int
    var array: NDArray

    init(
        name: String,
        scalarType: NDArray.ScalarType,
        sequenceDim: Int?,
        shape: [Int],
        layout: @escaping @Sendable ([Int]) -> (strides: [Int], byteCount: Int)
    ) {
        self.name = name
        self.scalarType = scalarType
        self.sequenceDim = sequenceDim
        self.layout = layout
        self.shape = shape
        (strides, byteCount) = layout(shape)
        array = NDArray(shape: shape, scalarType: scalarType, strides: strides)
        if sequenceDim == nil { zero() }
    }

    init(name: String, descriptor: NDArrayDescriptor, capacity: Int) throws {
        let dynamic = descriptor.shape.indices.filter { descriptor.shape[$0] < 0 }
        guard dynamic.count <= 1, dynamic.isEmpty || Self.elementSize(descriptor.scalarType) != nil else {
            throw EngineError.unsupportedState(name)
        }
        self.init(
            name: name,
            scalarType: descriptor.scalarType,
            sequenceDim: dynamic.first,
            shape: descriptor.shape.map { $0 < 0 ? capacity : $0 }
        ) { shape in
            let resolved = descriptor.resolvingDynamicDimensions(shape)
            return (resolved.preferredStrides, resolved.minimumByteCount)
        }
    }

    static func elementSize(_ type: NDArray.ScalarType) -> Int? {
        switch type {
        case .float16, .bfloat16: 2
        case .float32: 4
        default: nil
        }
    }

    func withBytes<R>(_ body: (UnsafeRawPointer) -> R) -> R {
        array.rawView().withUnsafeBytes { pointer, _, _ in body(pointer) }
    }

    mutating func withMutableBytes<R>(_ body: (UnsafeMutableRawPointer) -> R) -> R {
        array.mutableRawView().withUnsafeMutableBytes { pointer, _, _ in body(pointer) }
    }

    mutating func zero() {
        let count = byteCount
        withMutableBytes { _ = memset($0, 0, count) }
    }

    mutating func grow(to capacity: Int) {
        guard let dim = sequenceDim, let width = Self.elementSize(scalarType) else { return }
        var newShape = shape
        newShape[dim] = capacity
        let (newStrides, newByteCount) = layout(newShape)
        var newArray = NDArray(shape: newShape, scalarType: scalarType, strides: newStrides)
        let blockBytes = shape[dim] * strides[dim] * width
        let outer = Array(shape[..<dim])
        let oldStrides = strides
        withBytes { source in
            newArray.mutableRawView().withUnsafeMutableBytes { target, _, _ in
                var index = [Int](repeating: 0, count: dim)
                for _ in 0..<outer.reduce(1, *) {
                    let from = zip(index, oldStrides).reduce(0) { $0 + $1.0 * $1.1 } * width
                    let to = zip(index, newStrides).reduce(0) { $0 + $1.0 * $1.1 } * width
                    (target + to).copyMemory(from: source + from, byteCount: blockBytes)
                    var d = dim - 1
                    while d >= 0 {
                        index[d] += 1
                        if index[d] < outer[d] { break }
                        index[d] = 0
                        d -= 1
                    }
                }
            }
        }
        shape = newShape
        strides = newStrides
        byteCount = newByteCount
        array = newArray
    }
}

struct Context {
    var buffers: [StateBuffer]
    var capacity: Int
    let limit: Int
    var tokens: [Int32] = []

    func checkRoom(_ count: Int) throws {
        guard tokens.count + count <= limit else {
            throw EngineError.contextOverflow(needed: tokens.count + count, limit: limit)
        }
    }

    mutating func reserve(_ count: Int) {
        guard count > capacity else { return }
        let grown = Engine.grownCapacity(current: capacity, needed: count, limit: limit)
        for index in buffers.indices { buffers[index].grow(to: grown) }
        capacity = grown
    }

    func checkpoint() -> Engine.Checkpoint {
        Engine.Checkpoint(
            tokens: tokens,
            fixedStates: buffers.filter { $0.sequenceDim == nil }.map { state in
                state.withBytes { Data(bytes: $0, count: state.byteCount) }
            })
    }

    mutating func restore(_ checkpoint: Engine.Checkpoint) throws {
        let fixed = buffers.indices.filter { buffers[$0].sequenceDim == nil }
        guard tokens.starts(with: checkpoint.tokens),
            fixed.map({ buffers[$0].byteCount }) == checkpoint.fixedStates.map(\.count)
        else { throw EngineError.checkpointMismatch }
        for (index, data) in zip(fixed, checkpoint.fixedStates) {
            buffers[index].withMutableBytes { target in
                data.withUnsafeBytes { target.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            }
        }
        tokens = checkpoint.tokens
    }

    mutating func reset() {
        tokens = []
        for index in buffers.indices where buffers[index].sequenceDim == nil { buffers[index].zero() }
    }
}
