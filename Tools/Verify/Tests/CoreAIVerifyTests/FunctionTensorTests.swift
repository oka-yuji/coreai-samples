import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct FunctionTensorTests {
    @Test(
        "A layout is row-major when each stride is the product of the dimensions after it, and a dimension of 1 can have any stride",
        arguments: [
            ([1, 7], [7, 1]), ([2, 3], [3, 1]), ([1, 2, 5, 3], [99, 15, 3, 1]), ([2, 1, 3], [3, 99, 1]),
            ([1, 1], [0, 0]),
            ([4], [1]), ([1], [5]),
        ] as [([Int], [Int])])
    func recognizesRowMajor(shape: [Int], strides: [Int]) {
        #expect(FunctionTensor.hasRowMajorLayout(shape: shape, strides: strides))
    }

    @Test(
        "A layout with a transposed, padded or reversed stride is not row-major",
        arguments: [
            ([2, 3], [1, 2]), ([2, 3], [4, 1]), ([2, 3], [3, 2]), ([2, 3], [-3, 1]), ([3], [2]), ([1, 7], [7, 2]),
            ([2, 2, 2], [4, 1, 2]),
        ] as [([Int], [Int])])
    func rejectsOtherLayouts(shape: [Int], strides: [Int]) {
        #expect(!FunctionTensor.hasRowMajorLayout(shape: shape, strides: strides))
    }

    @Test("A layout whose strides do not match the number of dimensions is not row-major")
    func rejectsAMismatchedCount() {
        #expect(!FunctionTensor.hasRowMajorLayout(shape: [2, 3], strides: [3]))
        #expect(!FunctionTensor.hasRowMajorLayout(shape: [2], strides: [1, 1]))
    }
}
