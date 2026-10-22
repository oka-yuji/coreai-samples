import CoreAI
import ImageGenerationCore
import Testing

@testable import ImageGenerationRuntime

struct ImageCoreAIBufferTests {
    @Test(
        "Logical values survive padded and permuted Core AI storage without returning an aliased output",
        arguments: [[3, 1], [5, 1], [1, 2]])
    func floatStorage(_ strides: [Int]) throws {
        let contract = ImageTensorContract(name: "value", shape: [2, 3], dtype: .float32, layout: "row-major")
        var array = try ImageCoreAIBuffers.make(.floats([1, 2, 3, 4, 5, 6]), contract: contract, strides: strides)
        let copied = try ImageCoreAIBuffers.floats(array, contract: contract)
        #expect(copied == [1, 2, 3, 4, 5, 6])
        array.mutableRawView().withUnsafeMutableBytes { pointer, _, _ in
            pointer.storeBytes(of: Float(99), as: Float.self)
        }
        #expect(copied[0] == 1)
        #expect(try ImageCoreAIBuffers.floats(array, contract: contract)[0] == 99)
    }

    @Test("Integer tokens and boolean masks retain their physical storage representation")
    func integerAndMaskStorage() throws {
        let integer = ImageTensorContract(name: "ids", shape: [2, 2], dtype: .int32, layout: "row-major")
        let values: [Int32] = [0, 151_655, -1, .max]
        let array = try ImageCoreAIBuffers.make(.integers(values), contract: integer, strides: [3, 1])
        #expect(array.scalarType == .int32)
        let actual = array.rawView().withUnsafeBytes { pointer, _, _ in
            [0, 1, 3, 4].map { pointer.load(fromByteOffset: $0 * 4, as: Int32.self) }
        }
        #expect(actual == values)
        let mask = ImageTensorContract(name: "mask", shape: [2, 2], dtype: .bool, layout: "row-major")
        let flags = try ImageCoreAIBuffers.make(.mask([0, 1, 1, 0]), contract: mask, strides: [3, 1])
        #expect(flags.scalarType == .bool)
        let raw = flags.rawView().withUnsafeBytes { pointer, _, _ in
            [0, 1, 3, 4].map { pointer.load(fromByteOffset: $0, as: UInt8.self) }
        }
        #expect(raw == [0, 1, 1, 0])
    }

    @Test(
        "Overlapping negative incomplete and overflowing input strides fail before allocation",
        arguments: [[1, 1], [0, 1], [-1, 1], [3], [Int.max, 1]])
    func invalidStrides(_ strides: [Int]) throws {
        let contract = ImageTensorContract(name: "x", shape: [2, 3], dtype: .float32, layout: "row-major")
        #expect(throws: ImageGenerationError.invalidShape) {
            try ImageCoreAIBuffers.make(.floats([1, 2, 3, 4, 5, 6]), contract: contract, strides: strides)
        }
    }

    @Test("Nonfinite native output remains available for diagnostics before validation rejects it")
    func nonfiniteOutput() throws {
        let contract = ImageTensorContract(name: "x", shape: [1], dtype: .float32, layout: "row-major")
        var array = NDArray(shape: [1], scalarType: .float32)
        array.mutableRawView().withUnsafeMutableBytes { pointer, _, _ in
            pointer.storeBytes(of: Float.nan, as: Float.self)
        }
        let copied = try ImageCoreAIBuffers.floats(array, contract: contract)
        #expect(copied[0].isNaN)
        #expect(throws: ImageExecutionError.invalidTensor("x")) {
            try ImageTensorData.floats(copied).validate(contract)
        }
    }
}
