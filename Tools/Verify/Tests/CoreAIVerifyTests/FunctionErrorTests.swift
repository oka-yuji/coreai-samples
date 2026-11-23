import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct FunctionErrorTests {
    @Test(
        "Each rejection says what was wrong with the names and numbers involved",
        arguments: [
            (FunctionError.outputExists("out"), ["out", "exists"]),
            (.cannot("read a.bin", "No such file"), ["read a.bin", "No such file"]),
            (.invalidList("inputs[2].shape is bad"), ["inputs[2].shape is bad"]),
            (.noSuchFunction("draft", available: ["a", "b"]), ["draft", "a", "b"]),
            (.hasStates("draft", ["cache"]), ["draft", "cache", "state"]),
            (.duplicateInput("ids"), ["ids", "more than once"]),
            (.unknownInput("mask"), ["mask"]),
            (.missingInput("keys"), ["keys"]),
            (.unreadableType(input: "image", typeName: "Image (BGRA)"), ["image", "Image (BGRA)"]),
            (.dtypeMismatch(input: "keys", asset: "Float16", given: .float32), ["keys", "Float16", "float32"]),
            (.rankMismatch(input: "keys", asset: 4, given: 3), ["keys", "4", "3"]),
            (.dimensionMismatch(input: "keys", axis: 3, asset: 7, given: 8), ["keys", "3", "7", "8"]),
            (.dimensionMismatch(input: "keys", axis: 2, asset: -1, given: 0), ["keys", "2", "0", "1"]),
            (.notRegularFile("a.bin"), ["a.bin", "regular file"]),
            (.fileSize("a.bin", expected: 24, actual: 20), ["a.bin", "24", "20"]),
            (.unusableOutputName("a/b"), ["a/b"]),
        ] as [(FunctionError, [String])])
    func describes(error: FunctionError, words: [String]) {
        for word in words { #expect(error.description.contains(word)) }
    }
}
