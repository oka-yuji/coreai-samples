import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct FunctionTensorTypeTests {
    @Test(
        "The type text of an NDArray gives its dtype and its dimensions, with a question mark for a dynamic dimension",
        arguments: [
            ("NDArray (Int32, 1 \u{D7} 1)", "Int32", [1, 1]),
            ("NDArray (Float16, 1 \u{D7} 1 \u{D7} 48)", "Float16", [1, 1, 48]),
            ("NDArray (Float16, 1 \u{D7} 1 \u{D7} 2 \u{D7} ? \u{D7} 16)", "Float16", [1, 1, 2, -1, 16]),
            ("NDArray (Float32, ?)", "Float32", [-1]),
            ("NDArray (Float32, 7)", "Float32", [7]),
            ("NDArray (BFloat16, 0 \u{D7} 12)", "BFloat16", [0, 12]),
            ("NDArray (Int32,1\u{D7}1)", "Int32", [1, 1]),
        ] as [(String, String, [Int])])
    func readsTypes(text: String, dtype: String, dimensions: [Int]) throws {
        let type = try #require(FunctionTensorType(typeName: text))
        #expect(type.dtype == dtype)
        #expect(type.dimensions == dimensions)
    }

    @Test(
        "A text that is not an NDArray with a dtype and whole or question-mark dimensions is not read",
        arguments: [
            "", "NDArray", "NDArray ()", "NDArray (Int32)", "NDArray (Int32 1 \u{D7} 1)", "NDArray (Int32, 1 \u{D7} 1",
            "NDArray Int32, 1 \u{D7} 1)", "Image (BGRA, 1 \u{D7} 1)", "NDArray (Int32, 1 \u{D7} \u{D7} 2)",
            "NDArray (Int32, 1 \u{D7})", "NDArray (Int32, \u{D7} 1)", "NDArray (Int32, 1 x 1)",
            "NDArray (Int32, 1 \u{D7} a)",
            "NDArray (Int32, 1 \u{D7} -1)", "NDArray (Int32, 1 \u{D7} +1)", "NDArray (Int32, 1 \u{D7} 1.5)",
            "NDArray (Int32, 1 \u{D7} ??)",
        ])
    func rejectsOtherTexts(text: String) {
        #expect(FunctionTensorType(typeName: text) == nil)
    }
}
