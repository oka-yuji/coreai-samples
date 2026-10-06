import CoreAI
import Foundation
import GLiNERCore

public enum GLiNERModelError: Error, Equatable {
    case missingFunction(String)
    /// An input or output differs from the contract: missing or extra, not Int32 / Float32 (FP32), or another shape.
    case signatureMismatch(function: String, value: String)
    /// A run returned an output that is missing or not a contiguous Float32 array.
    case unreadableOutput(String)
}

/// The four Core AI functions of a GLiNER asset, specialized for the GPU, with the host steps between them.
///
/// Requests run one at a time in arrival order. A request cancelled while it waits, or between its function calls,
/// throws `CancellationError`; a function call in flight runs to its end.
public actor GLiNERModel {
    private let functions: [String: InferenceFunction]
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    /// Specializes the model for the GPU (or takes the cached specialization), checks every function against the
    /// contract before loading any, then loads the four functions.
    public init(asset: GLiNERAsset) async throws {
        let model = try await AIModel.specialize(
            contentsOf: asset.modelURL, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        for (name, signature) in Contract.implemented.functions {
            guard let descriptor = model.functionDescriptor(for: name) else {
                throw GLiNERModelError.missingFunction(name)
            }
            try Self.check(descriptor, of: name, against: signature)
        }
        var functions: [String: InferenceFunction] = [:]
        for name in Contract.implemented.functions.keys {
            guard let function = try model.loadFunction(named: name) else {
                throw GLiNERModelError.missingFunction(name)
            }
            functions[name] = function
        }
        self.functions = functions
    }

    /// Encodes an entity input, picks the candidates, scores them and decodes the spans of `text`.
    func extract(_ input: PreparedInput, text: String) async throws -> ExtractionResult {
        await lock()
        defer { unlock() }
        try Task.checkCancellation()
        let capacity = input.tokenCapacity
        let inputs = Self.encodeInputs(input)
        var encoded = try await run("encode_t\(capacity)", inputs)
        var outputs: [String: NDArray] = [:]
        for name in Contract.encodeOutputs {
            guard let output = encoded.remove(name)?.ndArray else { throw GLiNERModelError.unreadableOutput(name) }
            outputs[name] = output
        }
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
        pairInputs["cand_prior"] = NDArray(scalars: pool.priors, shape: [1, CandidatePool.size])
        var paired = try await run("pair_t\(capacity)", pairInputs)
        return try ExtractionResult(
            pairLogits: Self.matrix(paired.remove("pair_logits")?.ndArray, "pair_logits"),
            nullLogits: Self.floats(outputs["null_logits"], "null_logits"), pool: pool, input: input, text: text)
    }

    /// Encodes a classification input and returns the softmax of its class logits.
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

    /// Hands the lock to the next request still holding `busy`, so no request can slip in between.
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

    /// The values of a contiguous Float32 array (`run` returns row-major contiguous arrays).
    private static func floats(_ array: NDArray?, _ name: String) throws -> [Float] {
        guard let array, array.scalarType == .float32,
            let values = array.view(as: Float.self).contiguousElements?.withUnsafeBufferPointer({ Array($0) })
        else { throw GLiNERModelError.unreadableOutput(name) }
        return values
    }

    /// A `[1, rows, columns]` output without its batch axis.
    private static func matrix(_ array: NDArray?, _ name: String) throws -> Matrix {
        let values = try floats(array, name)
        guard let shape = array?.shape, shape.count == 3, shape[0] == 1 else {
            throw GLiNERModelError.unreadableOutput(name)
        }
        return Matrix(rows: shape[1], columns: shape[2], values: values)
    }

    private static let integerValues = Set(Contract.encodeInputs + ["cand_start", "cand_end", "cand_mask"])

    private static func check(
        _ descriptor: InferenceFunctionDescriptor, of name: String, against signature: Contract.Function
    ) throws {
        guard Set(descriptor.inputNames) == Set(signature.inputs), Set(descriptor.outputNames) == Set(signature.outputs)
        else { throw GLiNERModelError.signatureMismatch(function: name, value: "names") }
        let capacity = PreparedInput.tokenCapacities.first { name.hasSuffix("_t\($0)") } ?? 0
        let values =
            signature.inputs.map { ($0, descriptor.inputDescriptor(of: $0)) }
            + signature.outputs.map { ($0, descriptor.outputDescriptor(of: $0)) }
        for (value, valueDescriptor) in values {
            guard case .ndArray(let array)? = valueDescriptor,
                array.scalarType == (integerValues.contains(value) ? .int32 : .float32),
                matches(array.shape, shape(of: value, capacity: capacity))
            else { throw GLiNERModelError.signatureMismatch(function: name, value: value) }
        }
    }

    /// The shapes of the values the host writes or reads; `nil` dimensions are not checked.
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
