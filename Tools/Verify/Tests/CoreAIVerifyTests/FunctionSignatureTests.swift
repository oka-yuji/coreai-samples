import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct FunctionSignatureTests {
    private let signature = FunctionFixture.signature
    private let inputs = FunctionFixture.inputs

    private func replacing(_ name: String, shape: [Int]? = nil, dtype: FunctionDtype? = nil) -> [FunctionInput] {
        inputs.map {
            $0.name == name
                ? FunctionInput(name: name, dtype: dtype ?? $0.dtype, shape: shape ?? $0.shape, file: $0.file) : $0
        }
    }

    @Test("A list that gives every input with the dtype, the rank and the fixed dimensions of the asset is accepted")
    func acceptsAMatchingList() throws {
        try signature.check(inputs)
        try signature.check(inputs.reversed())
    }

    @Test(
        "A dynamic dimension is accepted at 1 and at a large size",
        arguments: [1, 2, 1_025, 16_896, Int.max])
    func acceptsADynamicDimension(size: Int) throws {
        try signature.check(replacing("keys", shape: [1, 2, size, 3]))
    }

    @Test("A dynamic dimension below 1 is rejected", arguments: [0, -1, Int.min])
    func rejectsADynamicDimensionBelowOne(size: Int) {
        #expect(throws: FunctionError.dimensionMismatch(input: "keys", axis: 2, asset: -1, given: size)) {
            try signature.check(replacing("keys", shape: [1, 2, size, 3]))
        }
    }

    @Test("A function with state is rejected whatever the list holds")
    func rejectsAFunctionWithState() {
        let stateful = FunctionSignature(name: "draft", inputs: signature.inputs, stateNames: ["cache", "other"])
        #expect(throws: FunctionError.hasStates("draft", ["cache", "other"])) { try stateful.check(inputs) }
        #expect(throws: FunctionError.hasStates("draft", ["cache", "other"])) { try stateful.check([]) }
    }

    @Test("An input that is listed twice is rejected")
    func rejectsADuplicateInput() {
        #expect(throws: FunctionError.duplicateInput("input_ids")) {
            try signature.check(inputs + [inputs[0]])
        }
    }

    @Test("An input that the function does not have is rejected")
    func rejectsAnUnknownInput() {
        let extra = FunctionInput(name: "mask", dtype: .int32, shape: [1], file: "mask.bin")
        #expect(throws: FunctionError.unknownInput("mask")) { try signature.check(inputs + [extra]) }
    }

    @Test("An input of the function that the list leaves out is rejected, with the first one in the order of the asset")
    func rejectsAMissingInput() {
        #expect(throws: FunctionError.missingInput("keys")) { try signature.check(Array(inputs.dropLast())) }
        #expect(throws: FunctionError.missingInput("input_ids")) { try signature.check(Array(inputs.dropFirst())) }
        #expect(throws: FunctionError.missingInput("input_ids")) { try signature.check([]) }
    }

    @Test("An input whose dtype differs from the asset is rejected")
    func rejectsAnotherDtype() {
        #expect(throws: FunctionError.dtypeMismatch(input: "hidden_states", asset: "Float16", given: .float32)) {
            try signature.check(replacing("hidden_states", dtype: .float32))
        }
        #expect(throws: FunctionError.dtypeMismatch(input: "input_ids", asset: "Int32", given: .float16)) {
            try signature.check(replacing("input_ids", dtype: .float16))
        }
    }

    @Test("An asset dtype that the list cannot name never matches")
    func rejectsAnAssetDtypeOutsideTheList() {
        let other = FunctionSignature(
            name: "draft", inputs: [.init(name: "a", typeName: "NDArray (BFloat16, 2)")], stateNames: [])
        for dtype in FunctionDtype.allCases {
            #expect(throws: FunctionError.dtypeMismatch(input: "a", asset: "BFloat16", given: dtype)) {
                try other.check([FunctionInput(name: "a", dtype: dtype, shape: [2], file: "a")])
            }
        }
    }

    @Test(
        "An input with more or fewer dimensions than the asset is rejected",
        arguments: [([1, 2, 5], 3), ([1, 2, 5, 3, 1], 5), ([1], 1)])
    func rejectsAnotherRank(shape: [Int], given: Int) {
        #expect(throws: FunctionError.rankMismatch(input: "keys", asset: 4, given: given)) {
            try signature.check(replacing("keys", shape: shape))
        }
    }

    @Test(
        "A fixed dimension that differs from the asset is rejected, whichever side is larger and wherever it is",
        arguments: [
            ("input_ids", [1, 2], 1, 1, 2), ("input_ids", [2, 1], 0, 1, 2), ("hidden_states", [1, 1, 5], 2, 4, 5),
            ("hidden_states", [1, 1, 3], 2, 4, 3), ("keys", [2, 2, 5, 3], 0, 1, 2), ("keys", [1, 3, 5, 3], 1, 2, 3),
            ("keys", [1, 1, 5, 3], 1, 2, 1), ("keys", [1, 2, 5, 4], 3, 3, 4),
        ] as [(String, [Int], Int, Int, Int)])
    func rejectsAnotherFixedDimension(name: String, shape: [Int], axis: Int, asset: Int, given: Int) {
        #expect(throws: FunctionError.dimensionMismatch(input: name, axis: axis, asset: asset, given: given)) {
            try signature.check(replacing(name, shape: shape))
        }
    }

    @Test("An input whose type text cannot be read is rejected")
    func rejectsAnUnreadableType() {
        let unreadable = FunctionSignature(
            name: "draft", inputs: [.init(name: "a", typeName: "Image (BGRA, 2 \u{D7} 2)")], stateNames: [])
        #expect(throws: FunctionError.unreadableType(input: "a", typeName: "Image (BGRA, 2 \u{D7} 2)")) {
            try unreadable.check([FunctionInput(name: "a", dtype: .int32, shape: [2, 2], file: "a")])
        }
    }

    @Test("An unreadable type text of an input that the list leaves out is reported as the missing input")
    func reportsAMissingInputBeforeAnUnreadableType() {
        let unreadable = FunctionSignature(
            name: "draft", inputs: [.init(name: "a", typeName: "Image (BGRA, 2 \u{D7} 2)")], stateNames: [])
        #expect(throws: FunctionError.missingInput("a")) { try unreadable.check([]) }
    }

    @Test(
        "The checks run in the order of state, duplicates, unknown inputs, missing inputs, dtype, rank and dimensions")
    func checksInOrder() {
        let wrongBoth = replacing("keys", shape: [1, 2, 5], dtype: .float32)
        #expect(throws: FunctionError.dtypeMismatch(input: "keys", asset: "Float16", given: .float32)) {
            try signature.check(wrongBoth)
        }
        let wrongRankAndSize = replacing("keys", shape: [9, 2, 5])
        #expect(throws: FunctionError.rankMismatch(input: "keys", asset: 4, given: 3)) {
            try signature.check(wrongRankAndSize)
        }
        let duplicateAndWrong = replacing("keys", shape: [9, 2, 5, 3]) + [inputs[0]]
        #expect(throws: FunctionError.duplicateInput("input_ids")) { try signature.check(duplicateAndWrong) }
        let unknownAndMissing = inputs.dropFirst() + [FunctionInput(name: "mask", dtype: .int32, shape: [1], file: "m")]
        #expect(throws: FunctionError.unknownInput("mask")) { try signature.check(Array(unknownAndMissing)) }
        let stateful = FunctionSignature(name: "draft", inputs: signature.inputs, stateNames: ["cache"])
        #expect(throws: FunctionError.hasStates("draft", ["cache"])) { try stateful.check(wrongBoth + [inputs[0]]) }
    }

    @Test("A function is found by its name, and a missing one is rejected with the names that the asset has")
    func findsAFunction() throws {
        let other = FunctionSignature(name: "other", inputs: [], stateNames: [])
        #expect(try FunctionSignature.find("draft", in: [other, signature]) == signature)
        #expect(throws: FunctionError.noSuchFunction("Draft", available: ["other", "draft"])) {
            try FunctionSignature.find("Draft", in: [other, signature])
        }
        #expect(throws: FunctionError.noSuchFunction("draft", available: [])) {
            try FunctionSignature.find("draft", in: [])
        }
    }
}
