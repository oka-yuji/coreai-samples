import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct FunctionInputTests {
    private func list(_ entries: String...) -> Data {
        Data(#"{"inputs":[\#(entries.joined(separator: ","))]}"#.utf8)
    }

    private func entry(
        name: String = #""a""#, dtype: String = #""int32""#, shape: String = "[1,2]", file: String = #""a.bin""#
    ) -> String {
        #"{"name":\#(name),"dtype":\#(dtype),"shape":\#(shape),"file":\#(file)}"#
    }

    private func reason(_ data: Data) throws -> String {
        let error = #expect(throws: FunctionError.self) { try FunctionInput.read(data) }
        guard case .invalidList(let reason) = try #require(error) else {
            Issue.record("The error is not an invalid list: \(String(describing: error))")
            return ""
        }
        return reason
    }

    @Test("A list of inputs is read in order with the names, dtypes, shapes and files that it gives")
    func readsInputs() throws {
        let data = list(
            entry(name: #""input_ids""#, shape: "[1,1]", file: #""ids.bin""#),
            entry(name: #""keys""#, dtype: #""float16""#, shape: "[1,2,5,3]", file: #""nested/keys.bin""#),
            entry(name: #""scale""#, dtype: #""float32""#, shape: "[7]", file: #""scale.bin""#))
        #expect(
            try FunctionInput.read(data) == [
                FunctionInput(name: "input_ids", dtype: .int32, shape: [1, 1], file: "ids.bin"),
                FunctionInput(name: "keys", dtype: .float16, shape: [1, 2, 5, 3], file: "nested/keys.bin"),
                FunctionInput(name: "scale", dtype: .float32, shape: [7], file: "scale.bin"),
            ])
    }

    @Test("A list with no inputs is read as no inputs")
    func readsAnEmptyList() throws {
        #expect(try FunctionInput.read(Data(#"{"inputs":[]}"#.utf8)).isEmpty)
    }

    @Test(
        "Text that is not a JSON object, an object without exactly the key inputs, and an inputs value that is not an array are rejected",
        arguments: [
            ("", "the top level is not a JSON object"),
            (#"{"inputs":["#, "the top level is not a JSON object"),
            ("[]", "the top level is not a JSON object"),
            ("1", "the top level is not a JSON object"),
            ("{}", "the top level must hold exactly the key inputs"),
            (#"{"inputs":[],"extra":1}"#, "the top level must hold exactly the key inputs"),
            (#"{"input":[]}"#, "the top level must hold exactly the key inputs"),
            (#"{"inputs":{}}"#, "inputs is not an array"),
            (#"{"inputs":null}"#, "inputs is not an array"),
        ])
    func rejectsAnotherTopLevel(text: String, message: String) throws {
        #expect(try reason(Data(text.utf8)) == message)
    }

    @Test(
        "An entry that is not an object, lacks one of the four keys or has another key is rejected, with its position",
        arguments: [
            (#"{"inputs":[1]}"#, "inputs[0] is not an object"),
            (#"{"inputs":["a"]}"#, "inputs[0] is not an object"),
            (#"{"inputs":[{"dtype":"int32","shape":[1],"file":"a"}]}"#, "inputs[0] must hold exactly the keys"),
            (#"{"inputs":[{"name":"a","shape":[1],"file":"a"}]}"#, "inputs[0] must hold exactly the keys"),
            (#"{"inputs":[{"name":"a","dtype":"int32","file":"a"}]}"#, "inputs[0] must hold exactly the keys"),
            (#"{"inputs":[{"name":"a","dtype":"int32","shape":[1]}]}"#, "inputs[0] must hold exactly the keys"),
            (
                #"{"inputs":[{"name":"a","dtype":"int32","shape":[1],"file":"a","unit":"x"}]}"#,
                "inputs[0] must hold exactly the keys"
            ),
        ])
    func rejectsABadEntry(text: String, message: String) throws {
        #expect(try reason(Data(text.utf8)).hasPrefix(message))
    }

    @Test("The position in the message is that of the first bad entry")
    func namesThePositionOfTheBadEntry() throws {
        #expect(try reason(list(entry(), entry(), #"{"name":"c"}"#)).hasPrefix("inputs[2] must hold exactly the keys"))
    }

    @Test(
        "A name that is not a string, a dtype that is not int32, float16 or float32, and a file that is not a string are rejected",
        arguments: [
            (#"{"name":1,"dtype":"int32","shape":[1],"file":"a"}"#, "inputs[0].name is not a string"),
            (#"{"name":null,"dtype":"int32","shape":[1],"file":"a"}"#, "inputs[0].name is not a string"),
            (
                #"{"name":"a","dtype":"int64","shape":[1],"file":"a"}"#,
                "inputs[0].dtype must be int32, float16 or float32"
            ),
            (
                #"{"name":"a","dtype":"Int32","shape":[1],"file":"a"}"#,
                "inputs[0].dtype must be int32, float16 or float32"
            ),
            (
                #"{"name":"a","dtype":"bfloat16","shape":[1],"file":"a"}"#,
                "inputs[0].dtype must be int32, float16 or float32"
            ),
            (#"{"name":"a","dtype":32,"shape":[1],"file":"a"}"#, "inputs[0].dtype must be int32, float16 or float32"),
            (#"{"name":"a","dtype":"int32","shape":[1],"file":1}"#, "inputs[0].file must be a relative path"),
            (#"{"name":"a","dtype":"int32","shape":[1],"file":null}"#, "inputs[0].file must be a relative path"),
        ])
    func rejectsABadScalar(entry: String, message: String) throws {
        #expect(try reason(Data(#"{"inputs":[\#(entry)]}"#.utf8)).hasPrefix(message))
    }

    @Test(
        "A shape that is empty, is not an array, or holds 0, a negative number, a fraction, a number written with a decimal point or an exponent, a boolean, a string, null or a number beyond the integer range is rejected",
        arguments: [
            "[]", "1", "null", "{}", "[0]", "[-1]", "[1,0]", "[2,-3]", "[-0]", "[1.5]", "[1.0]", "[1.00]", "[1e0]",
            "[1E0]",
            "[10e-1]", "[true]", "[false]", "[1,true]", #"["1"]"#, "[null]", "[[1]]", "[9223372036854775808]",
            "[100000000000000000000]",
        ])
    func rejectsABadShape(shape: String) throws {
        #expect(
            try reason(list(entry(shape: shape))).hasPrefix(
                "inputs[0].shape must be a non-empty array of integers from 1"))
    }

    @Test(
        "A shape of whole numbers from 1 is read, including 1 and the largest integer",
        arguments: [[1], [1, 1], [3, 20, 7], [Int.max]])
    func readsAShape(shape: [Int]) throws {
        let text = "[" + shape.map(String.init).joined(separator: ",") + "]"
        #expect(try FunctionInput.read(list(entry(shape: text))).map(\.shape) == [shape])
    }

    @Test(
        "A file that is empty, absolute, has a parent component or has a NUL character is rejected",
        arguments: [
            #""""#, #""/a.bin""#, #""/""#, #""../a.bin""#, #""a/../b.bin""#, #""a/..""#, #""..""#, #""a\u0000b""#,
        ])
    func rejectsAnUnsafeFile(file: String) throws {
        #expect(try reason(list(entry(file: file))).hasPrefix("inputs[0].file must be a relative path"))
    }

    @Test(
        "A file with a nested path, a current-folder component or dots inside a name is read",
        arguments: ["a.bin", "nested/deeper/a.bin", "./a.bin", "..a", "a..", "a..b/c", ".hidden"])
    func readsASafeFile(file: String) throws {
        #expect(try FunctionInput.read(list(entry(file: #""\#(file)""#))).map(\.file) == [file])
    }

    @Test("The byte count is the product of the shape and the width of the dtype")
    func countsBytes() {
        #expect(FunctionInput(name: "a", dtype: .int32, shape: [2, 3], file: "a").byteCount == 24)
        #expect(FunctionInput(name: "a", dtype: .float16, shape: [1, 1, 48], file: "a").byteCount == 96)
        #expect(FunctionInput(name: "a", dtype: .float32, shape: [5], file: "a").byteCount == 20)
        #expect(FunctionInput(name: "a", dtype: .float16, shape: [1], file: "a").byteCount == 2)
    }

    @Test("A byte count beyond the integer range stops at the largest integer, which no file can have")
    func capsTheByteCount() {
        #expect(FunctionInput(name: "a", dtype: .int32, shape: [Int.max, 2], file: "a").byteCount == Int.max)
        #expect(FunctionInput(name: "a", dtype: .int32, shape: [Int.max], file: "a").byteCount == Int.max)
        #expect(FunctionInput(name: "a", dtype: .float16, shape: [1 << 62, 1 << 62], file: "a").byteCount == Int.max)
    }
}
