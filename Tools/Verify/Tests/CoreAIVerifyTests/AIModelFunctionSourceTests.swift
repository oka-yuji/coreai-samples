import CoreAI
import Darwin
import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct AIModelFunctionSourceTests {
    @Test("The specialization request is the automatic choice of compute units with frequent reshapes expected")
    func requestsLikeTheEngine() {
        var expected = SpecializationOptions.default
        expected.expectFrequentReshapes = true
        let options = AIModelFunctionSource.specializationOptions()
        #expect(options == expected)
        #expect(options.preferredComputeUnitKind == nil)
        #expect(options.expectFrequentReshapes)
    }

    @Test(
        "The requested compute units are named default, cpu, gpu and neuralEngine",
        arguments: [(nil, "default"), (ComputeUnitKind.cpu, "cpu"), (.gpu, "gpu"), (.neuralEngine, "neuralEngine")]
            as [(ComputeUnitKind?, String)])
    func namesComputeUnits(kind: ComputeUnitKind?, name: String) {
        #expect(AIModelFunctionSource.computeUnitName(kind) == name)
    }

    @Test("The dtypes of the list and the scalar types of the SDK map to each other, and no other scalar type maps")
    func mapsScalarTypes() {
        for dtype in FunctionDtype.allCases { #expect(FunctionDtype(dtype.scalarType) == dtype) }
        #expect(FunctionDtype.allCases.map(\.scalarType) == [.int32, .float16, .float32])
        for scalarType in NDArray.ScalarType.allCases where ![.int32, .float16, .float32].contains(scalarType) {
            #expect(FunctionDtype(scalarType) == nil)
        }
    }

    @Test(
        "A tensor becomes an NDArray of the same dtype, shape and bytes, and the NDArray becomes the same tensor",
        arguments: [
            (FunctionDtype.int32, [1, 3]), (.float16, [1, 2, 3]), (.float32, [2, 1, 2]), (.float16, [1, 1, 2, 5, 16]),
        ])
    func roundTripsATensor(dtype: FunctionDtype, shape: [Int]) throws {
        let count = shape.reduce(1, *) * dtype.width
        let tensor = FunctionTensor(
            name: "t", dtype: dtype, shape: shape,
            data: Data((0..<count).map { UInt8(truncatingIfNeeded: $0 &* 7 &+ 1) }))
        let array = try AIModelFunctionSource.array(tensor)
        #expect(array.scalarType == dtype.scalarType)
        #expect(array.shape == shape)
        #expect(FunctionTensor.hasRowMajorLayout(shape: array.shape, strides: array.strides))
        #expect(try AIModelFunctionSource.tensor("t", array) == tensor)
    }

    @Test("An NDArray that is not stored row-major is rejected instead of being written in the wrong order")
    func rejectsAnotherLayout() {
        let array = NDArray(shape: [2, 3], scalarType: .float16, strides: [1, 2])
        #expect(throws: FunctionError.cannot("read the output t", "the array is not stored row-major")) {
            try AIModelFunctionSource.tensor("t", array)
        }
    }

    @Test(
        "An NDArray of a dtype that the list cannot name is rejected",
        arguments: [NDArray.ScalarType.bfloat16, .int64, .uint8, .float64])
    func rejectsAnotherScalarType(scalarType: NDArray.ScalarType) {
        let array = NDArray(shape: [2], scalarType: scalarType)
        #expect(throws: FunctionError.cannot("read the output t", "its dtype is not int32, float16 or float32")) {
            try AIModelFunctionSource.tensor("t", array)
        }
    }

    @Test(
        "An NDArray made for an input is row-major with no interleave layout, in the shapes of the inputs of a function",
        arguments: FunctionDtype.allCases,
        [
            [1, 1], [1, 1, 48], [1, 2, 3, 4], [1, 16, 1025, 8], [1, 1, 2, 5, 16], [1, 1, 2, 1025, 16],
            [1, 1, 1, 2000, 32],
        ])
    func makesRowMajorArrays(dtype: FunctionDtype, shape: [Int]) {
        let array = NDArray(shape: shape, scalarType: dtype.scalarType)
        #expect(array.interleaveLayout == nil)
        #expect(FunctionTensor.hasRowMajorLayout(shape: array.shape, strides: array.strides))
    }

    @Test(
        "The main hash is read in hex from a regular file, and a missing file, a folder, a link or a named pipe gives none without waiting",
        arguments: ["regular", "missing", "folder", "link", "pipe"])
    func readsTheMainHash(kind: String) throws {
        let base = FileManager.default.temporaryDirectory.appending(path: "main-hash-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: base) }
        let file = base.appending(path: "main.hash")
        let bytes = Data(0..<32)
        switch kind {
        case "regular": try bytes.write(to: file)
        case "folder": try FileManager.default.createDirectory(at: file, withIntermediateDirectories: false)
        case "link":
            try bytes.write(to: base.appending(path: "target"))
            try FileManager.default.createSymbolicLink(at: file, withDestinationURL: base.appending(path: "target"))
        case "pipe": try #require(mkfifo(file.path(percentEncoded: false), 0o600) == 0)
        default: break
        }
        let expected = kind == "regular" ? bytes.map { String(format: "%02x", $0) }.joined() : nil
        #expect(AIModelFunctionSource.mainHash(in: base) == expected)
    }
}
