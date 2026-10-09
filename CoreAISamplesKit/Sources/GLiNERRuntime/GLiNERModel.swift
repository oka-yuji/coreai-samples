import CoreAI
import Foundation
import GLiNERCore

public enum GLiNERModelError: Error, Equatable {
    case missingFunction(String)

    case signatureMismatch(function: String, value: String)

    case unreadableOutput(String)
}

public actor GLiNERModel {

    public nonisolated let floatType: NDArray.ScalarType
    private let functions: [String: InferenceFunction]
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    public init(
        asset: GLiNERAsset, options: SpecializationOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)
    ) async throws {
        let model = try await AIModel.specialize(contentsOf: asset.modelURL, options: options)
        var floatType: NDArray.ScalarType?
        for (name, signature) in Contract.implemented.functions.sorted(by: { $0.key < $1.key }) {
            guard let descriptor = model.functionDescriptor(for: name) else {
                throw GLiNERModelError.missingFunction(name)
            }
            floatType = try Self.check(
                inputs: Self.arrays(descriptor.inputNames, of: name, descriptor.inputDescriptor(of:)),
                outputs: Self.arrays(descriptor.outputNames, of: name, descriptor.outputDescriptor(of:)), of: name,
                against: signature, floatType: floatType)
        }
        var functions: [String: InferenceFunction] = [:]
        for name in Contract.implemented.functions.keys {
            guard let function = try model.loadFunction(named: name) else {
                throw GLiNERModelError.missingFunction(name)
            }
            functions[name] = function
        }
        self.functions = functions

        self.floatType = floatType ?? .float32
    }

    func extract(
        _ input: PreparedInput, text: String
    ) async throws -> (result: ExtractionResult, encode: Duration, pair: Duration) {
        await lock()
        defer { unlock() }
        try Task.checkCancellation()
        let clock = ContinuousClock()
        let started = clock.now
        let capacity = input.tokenCapacity
        let inputs = Self.encodeInputs(input)
        var encoded = try await run("encode_t\(capacity)", inputs)
        var outputs: [String: NDArray] = [:]
        for name in Contract.encodeOutputs {
            guard let output = encoded.remove(name)?.ndArray else { throw GLiNERModelError.unreadableOutput(name) }
            outputs[name] = output
        }
        let encodedAt = clock.now
        let pool = try CandidatePool(
            startLogits: Self.matrix(outputs["start_logits"], "start_logits"),
            endLogits: Self.matrix(outputs["end_logits"], "end_logits"),
            poolStart: Self.matrix(outputs["pool_start"], "pool_start"),
            poolEnd: Self.matrix(outputs["pool_end"], "pool_end"), wordCount: input.words.count,
            labelCount: input.labelCount)
        try Task.checkCancellation()

        var pairInputs = outputs.filter { Contract.pairEncodeInputs.contains($0.key) }
        pairInputs["word_mask"] = inputs["word_mask"]
        pairInputs["query_mask"] = inputs["query_mask"]
        pairInputs["cand_start"] = NDArray(scalars: pool.starts, shape: [1, CandidatePool.size])
        pairInputs["cand_end"] = NDArray(scalars: pool.ends, shape: [1, CandidatePool.size])
        pairInputs["cand_mask"] = NDArray(scalars: pool.mask, shape: [1, CandidatePool.size])
        pairInputs["cand_prior"] = Self.array(pool.priors, shape: [1, CandidatePool.size], as: floatType)
        var paired = try await run("pair_t\(capacity)", pairInputs)
        let result = try ExtractionResult(
            pairLogits: Self.matrix(paired.remove("pair_logits")?.ndArray, "pair_logits"),
            nullLogits: Self.floats(outputs["null_logits"], "null_logits"), pool: pool, input: input, text: text)
        return (result, encodedAt - started, clock.now - encodedAt)
    }

    func classify(_ input: PreparedInput) async throws -> ClassificationResult {
        await lock()
        defer { unlock() }
        try Task.checkCancellation()
        var encoded = try await run("encode_t\(input.tokenCapacity)", Self.encodeInputs(input))
        let logits = try Self.floats(encoded.remove("cls_logits")?.ndArray, "cls_logits")
        return ClassificationResult(logits: Array(logits.prefix(input.labelCount)))
    }

    private func run(_ name: String, _ inputs: [String: NDArray]) async throws -> InferenceFunction.Outputs {
        guard let function = functions[name] else { throw GLiNERModelError.missingFunction(name) }
        return try await function.run(inputs: inputs)
    }

    private func lock() async {
        if busy {
            await withCheckedContinuation { waiting.append($0) }
        } else {
            busy = true
        }
    }

    private func unlock() {
        if waiting.isEmpty { busy = false } else { waiting.removeFirst().resume() }
    }

    private static func encodeInputs(_ input: PreparedInput) -> [String: NDArray] {
        let tokens = input.tokenCapacity
        let labels = PreparedInput.labelCapacity
        return [
            "input_ids": NDArray(scalars: input.inputIDs, shape: [1, tokens]),
            "attention_mask": NDArray(scalars: input.attentionMask, shape: [1, tokens]),
            "word_idx": NDArray(scalars: input.wordIndex, shape: [1, tokens]),
            "word_mask": NDArray(scalars: input.wordMask, shape: [1, tokens]),
            "query_idx": NDArray(scalars: input.queryIndex, shape: [1, labels]),
            "query_mask": NDArray(scalars: input.queryMask, shape: [1, labels]),
            "cls_idx": NDArray(scalars: input.classIndex, shape: [1, labels]),
            "cls_mask": NDArray(scalars: input.classMask, shape: [1, labels]),
        ]
    }

    static func floats(_ array: NDArray?, _ name: String) throws -> [Float] {
        guard let array else { throw GLiNERModelError.unreadableOutput(name) }
        let values: [Float]? =
            switch array.scalarType {
            case .float32: array.view(as: Float.self).contiguousElements?.withUnsafeBufferPointer { Array($0) }
            case .float16:
                array.view(as: Float16.self).contiguousElements?.withUnsafeBufferPointer { $0.map(Float.init) }
            default: nil
            }
        guard let values else { throw GLiNERModelError.unreadableOutput(name) }
        return values
    }

    static func matrix(_ array: NDArray?, _ name: String) throws -> Matrix {
        let values = try floats(array, name)
        guard let shape = array?.shape, shape.count == 3, shape[0] == 1 else {
            throw GLiNERModelError.unreadableOutput(name)
        }
        return Matrix(rows: shape[1], columns: shape[2], values: values)
    }

    static func array(_ values: [Float], shape: [Int], as type: NDArray.ScalarType) -> NDArray {
        type == .float16
            ? NDArray(scalars: values.map(Float16.init), shape: shape) : NDArray(scalars: values, shape: shape)
    }

    private static let integerValues = Set(Contract.encodeInputs + ["cand_start", "cand_end", "cand_mask"])

    typealias ArrayValue = (scalarType: NDArray.ScalarType, shape: [Int])

    private static func arrays(
        _ names: [String], of function: String, _ descriptor: (String) -> InferenceValue.Descriptor?
    ) throws -> [String: ArrayValue] {
        var arrays: [String: ArrayValue] = [:]
        for name in names {
            guard case .ndArray(let array)? = descriptor(name) else {
                throw GLiNERModelError.signatureMismatch(function: function, value: name)
            }
            arrays[name] = (array.scalarType, array.shape)
        }
        return arrays
    }

    static func check(
        inputs: [String: ArrayValue], outputs: [String: ArrayValue], of name: String,
        against signature: Contract.Function, floatType: NDArray.ScalarType?
    ) throws -> NDArray.ScalarType? {
        guard Set(inputs.keys) == Set(signature.inputs), Set(outputs.keys) == Set(signature.outputs)
        else { throw GLiNERModelError.signatureMismatch(function: name, value: "names") }
        let capacity = PreparedInput.tokenCapacities.first { name.hasSuffix("_t\($0)") } ?? 0
        let values = signature.inputs.map { ($0, inputs[$0]) } + signature.outputs.map { ($0, outputs[$0]) }
        var floatType = floatType
        for (value, array) in values {
            let integer = integerValues.contains(value)
            let allowed: [NDArray.ScalarType] = integer ? [.int32] : floatType.map { [$0] } ?? [.float32, .float16]
            guard let array, allowed.contains(array.scalarType),
                matches(array.shape, shape(of: value, capacity: capacity))
            else { throw GLiNERModelError.signatureMismatch(function: name, value: value) }
            if !integer { floatType = array.scalarType }
        }
        return floatType
    }

    private static func shape(of value: String, capacity: Int) -> [Int?]? {
        let labels = PreparedInput.labelCapacity
        let slots = CandidatePool.size
        switch value {
        case "input_ids", "attention_mask", "word_idx", "word_mask": return [1, capacity]
        case "query_idx", "query_mask", "cls_idx", "cls_mask", "null_logits", "cls_logits": return [1, labels]
        case "start_logits", "end_logits": return [1, labels, capacity + 1]
        case "pool_start", "pool_end": return [1, capacity + 1, nil]
        case "cand_start", "cand_end", "cand_mask", "cand_prior": return [1, slots]
        case "pair_logits": return [1, slots, labels]
        default: return nil
        }
    }

    private static func matches(_ shape: [Int], _ expected: [Int?]?) -> Bool {
        guard let expected else { return true }
        return shape.count == expected.count && zip(shape, expected).allSatisfy { $1 == nil || $0 == $1 }
    }
}
