import CoreAI
import GLiNERCore
import Testing

@testable import GLiNERRuntime

private typealias PairArrays = (inputs: [String: GLiNERModel.ArrayValue], outputs: [String: GLiNERModel.ArrayValue])

/// `pair_t256` as `coreai-build inspect` lists it for the GL4 exports, with `float` for every floating-point value.
private func pairArrays(float: NDArray.ScalarType) -> PairArrays {
    let inputs: [String: GLiNERModel.ArrayValue] = [
        "scorer_start": (float, [1, 257, 128]), "scorer_end": (float, [1, 257, 128]),
        "content_prefix": (float, [1, 257, 64]), "query_proj": (float, [1, 16, 128]),
        "start_logits": (float, [1, 16, 257]), "end_logits": (float, [1, 16, 257]),
        "inside_prefix": (float, [1, 16, 257]), "inside_mean": (float, [1, 16, 1]),
        "word_mask": (.int32, [1, 256]), "query_mask": (.int32, [1, 16]), "cand_start": (.int32, [1, 192]),
        "cand_end": (.int32, [1, 192]), "cand_mask": (.int32, [1, 192]), "cand_prior": (float, [1, 192]),
    ]
    return (inputs, ["pair_logits": (float, [1, 192, 16])])
}

/// Checks `arrays` as `pair_t256`, after functions whose floating-point values were of `floatType`.
private func checkPair(_ arrays: PairArrays, floatType: NDArray.ScalarType? = nil) throws -> NDArray.ScalarType? {
    try GLiNERModel.check(
        inputs: arrays.inputs, outputs: arrays.outputs, of: "pair_t256",
        against: #require(Contract.implemented.functions["pair_t256"]), floatType: floatType)
}

struct GLiNERModelTests {
    @Test("A function whose floating values are Float16 passes the signature check and its outputs are read as Float")
    func float16FunctionIsReadAsFloat() throws {
        #expect(try checkPair(pairArrays(float: .float16)) == .float16)
        #expect(try checkPair(pairArrays(float: .float16), floatType: .float16) == .float16)

        let values: [Float] = [0.5, -1.25, 3, 0, 65_504, -10_000]
        let logits = NDArray(scalars: values.map(Float16.init), shape: [1, 2, 3])
        #expect(try GLiNERModel.matrix(logits, "pair_logits") == Matrix(rows: 2, columns: 3, values: values))
        let prior = GLiNERModel.array([0.375, -2], shape: [1, 2], as: .float16)
        #expect(prior.scalarType == .float16)
        #expect(try GLiNERModel.floats(prior, "cand_prior") == [0.375, -2])
    }

    @Test(
        "A function whose floating values are neither Float32 nor Float16 is a signature mismatch",
        arguments: [NDArray.ScalarType.bfloat16, .float64])
    func otherFloatTypesAreRefused(type: NDArray.ScalarType) {
        #expect(throws: GLiNERModelError.signatureMismatch(function: "pair_t256", value: "scorer_start")) {
            try checkPair(pairArrays(float: type))
        }
    }

    @Test("Floating values of two types, in one function or after another function's, are a signature mismatch")
    func mixedFloatTypesAreRefused() {
        var mixed = pairArrays(float: .float16)
        mixed.inputs["cand_prior"] = (.float32, [1, 192])
        #expect(throws: GLiNERModelError.signatureMismatch(function: "pair_t256", value: "cand_prior")) {
            try checkPair(mixed)
        }
        #expect(throws: GLiNERModelError.signatureMismatch(function: "pair_t256", value: "scorer_start")) {
            try checkPair(pairArrays(float: .float32), floatType: .float16)
        }
    }
}
