import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct FunctionDtypeTests {
    @Test(
        "Each dtype has the name and the width in bytes of its elements",
        arguments: [(FunctionDtype.int32, "int32", 4), (.float16, "float16", 2), (.float32, "float32", 4)])
    func namesAndWidths(dtype: FunctionDtype, name: String, width: Int) {
        #expect(dtype.rawValue == name)
        #expect(dtype.width == width)
    }

    @Test("The type names of the asset are read as the dtypes of the list")
    func readsAssetNames() {
        #expect(FunctionDtype(assetName: "Int32") == .int32)
        #expect(FunctionDtype(assetName: "Float16") == .float16)
        #expect(FunctionDtype(assetName: "Float32") == .float32)
    }

    @Test(
        "Other type names of the asset are not read",
        arguments: ["", "BFloat16", "Int64", "Float64", "UInt8", "Bool", "int", "Float"])
    func rejectsOtherAssetNames(name: String) {
        #expect(FunctionDtype(assetName: name) == nil)
    }
}
