import CoreAI
import Foundation
import Testing

@testable import CoreAIBackend

struct TypedDecisionOutputTests {
    @Test(
        "Both output contracts read the final token from strided arrays at the input length boundaries",
        arguments: [1, 7, 8_192])
    func finalToken(_ tokens: Int) throws {
        let width = 4
        let expected = (0..<width).map { Float(((tokens - 1) % 100) * width + $0) }
        for output in [TypedDecisionOutput.allTokens, .lastToken] {
            let shape = output.shape(tokens: tokens, width: width)
            let rows = shape[1]
            var array = NDArray(shape: shape, scalarType: .float16, strides: [rows * 8, 8, 1])
            array.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, strides in
                pointer.update(repeating: -999, count: (rows - 1) * 8 + width)
                for row in 0..<rows {
                    let source = output == .allTokens ? row : tokens - 1
                    for column in 0..<width {
                        pointer[row * strides[1] + column * strides[2]] = Float16((source % 100) * width + column)
                    }
                }
            }
            #expect(try output.values(from: array, width: width).map(\.bitPattern) == expected.map(\.bitPattern))
        }
    }

    @Test("The descriptor contract distinguishes a dynamic sequence from exactly one output token")
    func descriptorContracts() {
        #expect(TypedDecisionOutput.allTokens.accepts(shape: [1, -1, 4], type: .float16, interleaved: false, width: 4))
        #expect(!TypedDecisionOutput.allTokens.accepts(shape: [1, 1, 4], type: .float16, interleaved: false, width: 4))
        #expect(TypedDecisionOutput.lastToken.accepts(shape: [1, 1, 4], type: .float16, interleaved: false, width: 4))
        for shape in [[1, -1, 4], [1, 2, 4], [1, 4], [2, 1, 4], [1, 1, 8]] {
            #expect(!TypedDecisionOutput.lastToken.accepts(shape: shape, type: .float16, interleaved: false, width: 4))
        }
        #expect(!TypedDecisionOutput.lastToken.accepts(shape: [1, 1, 4], type: .float32, interleaved: false, width: 4))
        #expect(!TypedDecisionOutput.lastToken.accepts(shape: [1, 1, 4], type: .float16, interleaved: true, width: 4))
    }

    @Test("The reader rejects a mismatched output layout or scalar type before reading memory")
    func invalidOutputs() {
        let sequence = NDArray(shape: [1, 2, 4], scalarType: .float16)
        let wrongType = NDArray(shape: [1, 1, 4], scalarType: .float32)
        #expect(throws: TypedDecisionError.self) { try TypedDecisionOutput.lastToken.values(from: sequence, width: 4) }
        #expect(throws: TypedDecisionError.self) { try TypedDecisionOutput.lastToken.values(from: wrongType, width: 4) }
        #expect(throws: TypedDecisionError.self) { try TypedDecisionOutput.allTokens.values(from: sequence, width: 8) }
    }

    @Test("The bundle accepts both declared output names and refuses an unknown output")
    func bundleOutputs() throws {
        let root = try TypedDecisionFakeBundle.create()
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        for output in [TypedDecisionOutput.allTokens, .lastToken] {
            let document = TypedDecisionFakeBundle.setting(
                output.rawValue, at: ["backbone", "output"], in: TypedDecisionFakeBundle.document())
            try TypedDecisionFakeBundle.write(document, to: root)
            #expect(try TypedDecisionBundle(url: root).backboneOutput == output.rawValue)
        }
        let invalid = TypedDecisionFakeBundle.setting(
            "unknown", at: ["backbone", "output"], in: TypedDecisionFakeBundle.document())
        try TypedDecisionFakeBundle.write(invalid, to: root)
        #expect(throws: TypedDecisionError.self) { try TypedDecisionBundle(url: root) }
    }
}
