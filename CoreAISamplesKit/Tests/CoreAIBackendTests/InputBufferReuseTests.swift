import CoreAI
import Testing

@testable import CoreAIBackend

@Suite struct InputBufferReuseTests {
    private struct Views: Equatable {
        var inputIDs: [[Int32]]
        var positionIDs: [[Int32]]
        var contiguous: Bool
    }

    private static func rows(_ view: borrowing NDArray.RawView) -> [[Int32]] {
        let shape = view.shape
        let strides = view.strides
        let (height, width, rowStride, columnStride) = (shape[0], shape[1], strides[0], strides[1])
        return view.withUnsafeBytes { pointer, _, _ in
            (0..<height).map { row in
                (0..<width).map { column in
                    pointer.load(fromByteOffset: (row * rowStride + column * columnStride) * 4, as: Int32.self)
                }
            }
        }
    }

    private static func address(_ array: NDArray) -> UInt {
        array.rawView().withUnsafeBytes { pointer, _, _ in UInt(bitPattern: pointer) }
    }

    private static func overwrite(_ array: inout NDArray, with value: Int32) {
        array.mutableView(as: Int32.self).withUnsafeMutablePointer { pointer, shape, _ in
            for index in 0..<shape[1] { pointer[index] = value }
        }
    }

    private static func forward(_ context: inout Context, _ chunk: [Int32]) -> Views {
        let end = context.tokens.count + chunk.count
        context.reserve(end)
        context.writeInputs(chunk[...])
        let inputTokens = context.inputTokens
        let positions = context.positions
        let views = Views(
            inputIDs: rows(inputTokens.rawView().slice(at: [0..<1, 0..<chunk.count])),
            positionIDs: rows(positions.rawView().slice(at: [0..<1, 0..<end])),
            contiguous: inputTokens.view(as: Int32.self).slice(at: [0..<1, 0..<chunk.count]).isContiguous
                && positions.view(as: Int32.self).slice(at: [0..<1, 0..<end]).isContiguous)
        context.tokens.append(contentsOf: chunk)
        return views
    }

    private static func reference(_ chunk: [Int32], after processed: Int) -> Views {
        Views(
            inputIDs: [chunk], positionIDs: [(0..<processed + chunk.count).map { Int32($0) }], contiguous: true)
    }

    @Test(
        "input_ids and the full-length position_ids match the reference across capacity boundaries and after reset and restore"
    )
    func inputsMatchTheReferenceAcrossCapacityBoundariesResetAndRestore() throws {
        var context = Context(buffers: [], capacity: 4, limit: 64)
        func check(_ chunk: [Int32], tokens: Int, positions: Int, _ label: String) {
            let expected = Self.reference(chunk, after: context.tokens.count)
            let actual = Self.forward(&context, chunk)
            #expect(actual == expected, "\(label)")
            #expect(context.inputTokens.shape == [1, tokens], "\(label)")
            #expect(context.positions.shape == [1, positions], "\(label)")
        }

        check([11, 12, 13, 14, 15], tokens: 8, positions: 8, "first prefill")
        let saved = context.checkpoint()
        check([16], tokens: 8, positions: 8, "decode within capacity")
        check([17], tokens: 8, positions: 8, "decode within capacity")
        check([18], tokens: 8, positions: 8, "decode filling capacity")
        check([19], tokens: 8, positions: 16, "decode crossing the position capacity")
        check(Array(20..<32), tokens: 16, positions: 32, "prefill crossing both capacities")
        check([32], tokens: 16, positions: 32, "decode after the prefill")

        try context.restore(saved)
        #expect(context.tokens == [11, 12, 13, 14, 15])
        check([40, 41, 42], tokens: 16, positions: 32, "prefill after restore")
        check([43], tokens: 16, positions: 32, "decode after restore")

        Self.overwrite(&context.positions, with: -1)
        context.reset()
        check([50, 51, 52, 53], tokens: 16, positions: 32, "prefill after reset")
        check([54], tokens: 16, positions: 32, "decode after reset")
        check(Array(55..<95), tokens: 64, positions: 64, "prefill crossing to the limit after reset")
    }

    @Test(
        "Repeated decodes within the same capacity reuse the input buffers without reallocating them, and unloading does not keep them"
    )
    func decodesReuseTheInputBuffersAndUnloadingDoesNotKeepThem() {
        var context = Context(buffers: [], capacity: 4, limit: 64)
        _ = Self.forward(&context, [1, 2, 3, 4, 5])
        let tokenAddress = Self.address(context.inputTokens)
        let positionAddress = Self.address(context.positions)
        for token in Int32(6)...8 {
            _ = Self.forward(&context, [token])
            #expect(Self.address(context.inputTokens) == tokenAddress, "decode \(token)")
            #expect(Self.address(context.positions) == positionAddress, "decode \(token)")
        }
        context.reset()
        _ = Self.forward(&context, [1, 2])
        #expect(Self.address(context.inputTokens) == tokenAddress)
        #expect(Self.address(context.positions) == positionAddress)

        var shared = context.positions
        Self.overwrite(&shared, with: 0)
        #expect(Self.address(shared) != positionAddress)
        #expect(Self.address(context.positions) == positionAddress)

        var inputTokens = context.inputTokens
        var positions = context.positions
        context.releaseInputs()
        #expect(context.inputTokens.shape == [1, 0])
        #expect(context.positions.shape == [1, 0])
        #expect(context.writtenPositions == 0)
        Self.overwrite(&inputTokens, with: 0)
        Self.overwrite(&positions, with: 0)
        #expect(Self.address(inputTokens) == tokenAddress)
        #expect(Self.address(positions) == positionAddress)
    }
}
