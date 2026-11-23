import Foundation

@testable import coreai_verify

struct FunctionFixture {
    static let signature = FunctionSignature(
        name: "draft",
        inputs: [
            .init(name: "input_ids", typeName: "NDArray (Int32, 1 \u{D7} 1)"),
            .init(name: "hidden_states", typeName: "NDArray (Float16, 1 \u{D7} 1 \u{D7} 4)"),
            .init(name: "keys", typeName: "NDArray (Float16, 1 \u{D7} 2 \u{D7} ? \u{D7} 3)"),
        ],
        stateNames: [])

    static let inputs = [
        FunctionInput(name: "input_ids", dtype: .int32, shape: [1, 1], file: "input_ids.bin"),
        FunctionInput(name: "hidden_states", dtype: .float16, shape: [1, 1, 4], file: "hidden_states.bin"),
        FunctionInput(name: "keys", dtype: .float16, shape: [1, 2, 5, 3], file: "keys.bin"),
    ]

    static func data(_ input: FunctionInput) -> Data {
        Data((0..<input.byteCount).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ input.name.utf8.count) })
    }

    static func integers(_ values: [Int32]) -> Data {
        values.reduce(into: Data()) { data, value in
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
    }

    static func outputs(token: Int32 = 1) -> [FunctionTensor] {
        [
            FunctionTensor(
                name: "draft_tokens", dtype: .int32, shape: [1, 3], data: integers([token, token + 1, token + 2])),
            FunctionTensor(name: "scores", dtype: .float16, shape: [1, 2], data: Data([1, 2, 3, 4])),
        ]
    }

    let folder: URL

    var list: URL { folder.appending(path: "inputs.json") }
    var output: URL { folder.appending(path: "out") }

    init(inputs: [FunctionInput] = Self.inputs, listText: String? = nil) throws {
        folder = FileManager.default.temporaryDirectory.appending(path: "function-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        for input in inputs { try Self.data(input).write(to: folder.appending(path: input.file)) }
        if let listText { try Data(listText.utf8).write(to: list) } else { try writeList(inputs) }
    }

    func writeList(_ inputs: [FunctionInput]) throws {
        let entries = inputs.map {
            ["name": $0.name, "dtype": $0.dtype.rawValue, "shape": $0.shape, "file": $0.file] as [String: Any]
        }
        try JSONSerialization.data(withJSONObject: ["inputs": entries]).write(to: list)
    }

    func remove() {
        try? FileManager.default.removeItem(at: folder)
    }

    func run(
        _ source: FakeFunctionSource, repeatCount: Int = 1, function: String = "draft", folderMode: mode_t = 0o755
    ) async throws -> FunctionRecord {
        try await FunctionRun.run(
            function: function, inputs: list, output: output, repeatCount: repeatCount, source: source,
            folderMode: folderMode)
    }

    static func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path(percentEncoded: false))
    }

    func names() throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: output.path(percentEncoded: false)).sorted()
    }

    func failureKind() throws -> String? {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: output.appending(path: "failure.json")))
        return (object as? [String: Any])?["kind"] as? String
    }

    func url(_ file: String) -> URL {
        folder.appending(path: file)
    }

    func contents(of file: String) throws -> Data {
        try Data(contentsOf: url(file))
    }
}
