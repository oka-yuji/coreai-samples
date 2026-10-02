import Foundation
import Testing

@testable import CoreAIBackend

@Test func readsMetadata() throws {
    let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let json = """
        {"metadata_version": "0.2", "kind": "llm", "name": "tiny",
         "assets": {"main": "tiny.aimodel"},
         "language": {"tokenizer": "x/tiny", "vocab_size": 151936, "max_context_length": 8192,
                      "embedded_tokenizer": true, "function_map": {"main": ["main"]}}}
        """
    try Data(json.utf8).write(to: folder.appending(path: "metadata.json"))

    let bundle = try ModelBundle(url: folder)
    #expect(bundle.name == "tiny")
    #expect(bundle.modelURL == folder.appending(path: "tiny.aimodel"))
    #expect(bundle.tokenizerURL == folder.appending(path: "tokenizer"))
    #expect(bundle.vocabSize == 151936)
    #expect(bundle.maxContextLength == 8192)

    try Data(#"{"name": "tiny", "assets": {"main": "tiny.aimodel"}}"#.utf8).write(to: folder.appending(path: "metadata.json"))
    #expect(throws: DecodingError.self) { try ModelBundle(url: folder) }
}

@Test func growsCapacityByDoublingUpToTheLimit() {
    #expect(Engine.grownCapacity(current: 256, needed: 256, limit: 8192) == 256)
    #expect(Engine.grownCapacity(current: 256, needed: 257, limit: 8192) == 512)
    #expect(Engine.grownCapacity(current: 256, needed: 3000, limit: 8192) == 4096)
    #expect(Engine.grownCapacity(current: 4096, needed: 5000, limit: 6000) == 6000)
    #expect(Engine.grownCapacity(current: 1, needed: 7, limit: 16384) == 8)
}

func contiguous(_ width: Int) -> @Sendable ([Int]) -> (strides: [Int], byteCount: Int) {
    { shape in
        let strides = shape.indices.map { shape[($0 + 1)...].reduce(1, *) }
        return (strides, shape.reduce(1, *) * width)
    }
}

@Test func growingKeepsEveryRow() {
    var state = StateBuffer(
        name: "keyCache", scalarType: .float16, sequenceDim: 3, shape: [2, 1, 3, 4, 5], layout: contiguous(2))
    state.withMutableBytes { bytes in
        for i in 0..<(2 * 3 * 4 * 5) { bytes.storeBytes(of: UInt16(i + 1), toByteOffset: i * 2, as: UInt16.self) }
    }

    state.grow(to: 8)

    #expect(state.shape == [2, 1, 3, 8, 5])
    #expect(state.strides == [120, 120, 40, 5, 1])
    #expect(state.byteCount == 2 * 3 * 8 * 5 * 2)
    state.withBytes { bytes in
        for block in 0..<6 {
            for row in 0..<4 {
                for column in 0..<5 {
                    let value = bytes.load(fromByteOffset: (block * 40 + row * 5 + column) * 2, as: UInt16.self)
                    #expect(value == UInt16(block * 20 + row * 5 + column + 1))
                }
            }
        }
    }
}

@Test func checkpointRestoresFixedStatesAndRejectsAnOverwrittenPrefix() throws {
    let kv = StateBuffer(name: "keyCache", scalarType: .bfloat16, sequenceDim: 1, shape: [2, 4, 3], layout: contiguous(2))
    let fixed = StateBuffer(
        name: "recurrentStates", scalarType: .bfloat16, sequenceDim: nil, shape: [2, 6], layout: contiguous(2))
    var context = Context(buffers: [kv, fixed], capacity: 4, limit: 64)
    func fill(_ value: Int32) {
        context.buffers[1].withMutableBytes { _ = memset($0, value, 24) }
    }
    func fixedBytes() -> [UInt8] {
        context.buffers[1].withBytes { Array(UnsafeRawBufferPointer(start: $0, count: 24)) }
    }
    #expect(fixedBytes() == Array(repeating: 0, count: 24))

    context.tokens = [1, 2, 3]
    fill(7)
    let saved = context.checkpoint()
    #expect(saved.tokens == [1, 2, 3])
    #expect(saved.stateBytes == 24)

    context.tokens += [4, 5]
    fill(9)
    try context.restore(saved)
    #expect(context.tokens == [1, 2, 3])
    #expect(fixedBytes() == Array(repeating: 7, count: 24))

    context.tokens = [1, 2, 8]
    #expect(throws: EngineError.checkpointMismatch) { try context.restore(saved) }

    context.reset()
    #expect(context.tokens.isEmpty)
    #expect(fixedBytes() == Array(repeating: 0, count: 24))
    #expect(throws: EngineError.checkpointMismatch) { try context.restore(saved) }

    var attentionOnly = Context(buffers: [kv], capacity: 4, limit: 64)
    attentionOnly.tokens = [5, 6, 7, 8]
    let point = attentionOnly.checkpoint()
    attentionOnly.tokens = [5, 6, 7, 8, 9, 10]
    try attentionOnly.restore(Engine.Checkpoint(tokens: [5, 6], fixedStates: []))
    #expect(attentionOnly.tokens == [5, 6])
    #expect(point.stateBytes == 0)
    #expect(throws: EngineError.checkpointMismatch) { try attentionOnly.restore(point) }
    #expect(throws: EngineError.contextOverflow(needed: 65, limit: 64)) { try attentionOnly.checkRoom(63) }
}

@Test func greedyPicksTheFirstMaximum() {
    func pick(_ values: [Float16]) -> Int32 {
        var scratch = [Float](repeating: 0, count: values.count)
        return values.withUnsafeBufferPointer { Engine.argmax($0, scratch: &scratch) }
    }
    #expect(pick([1, 3, 2]) == 1)
    #expect(pick([-3, -1, -2]) == 1)
    #expect(pick([0, 5, 5, 1]) == 1)
    #expect(pick([2, 0, 2]) == 0)

    var values = (0..<248_320).map { Float16(Float(($0 * 7919) % 1000) / 100) }
    values[1000] = 12
    values[200_000] = 12
    #expect(pick(values) == 1000)
}

let bundlePaths = (ProcessInfo.processInfo.environment["COREAI_TEST_BUNDLES"] ?? "")
    .split(separator: ":").map(String.init)

@Test(.enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
func modelRunsAndRestores(path: String) async throws {
    let bundle = try ModelBundle(url: URL(filePath: path))
    let tokenizer = try await bundle.loadTokenizer()
    let engine = try await Engine(bundle: bundle)
    let stop = Set([tokenizer.eosTokenId].compactMap { $0.map(Int32.init) })
    let prefix = tokenizer.encode(text: "The following is a short conversation about geography.\n").map(Int32.init)
    let suffix = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "Name three rivers in Europe."]])
        .map(Int32.init)

    try await engine.prefill(prefix)
    let point = engine.checkpoint()
    #expect((point.stateBytes > 0) == engine.isHybrid)
    let first = try await engine.generate(suffix, maxTokens: 24, stopTokens: stop)
    try engine.restore(point)
    let second = try await engine.generate(suffix, maxTokens: 24, stopTokens: stop)
    #expect(first == second)

    engine.reset()
    try await engine.prefill(prefix + suffix, usePrefillFunction: false)
    var viaMain = [engine.greedyToken()]
    while viaMain.count < 16 {
        try await engine.step(viaMain.last!)
        viaMain.append(engine.greedyToken())
    }
    #expect(viaMain.allSatisfy { (0..<Int32(engine.vocabSize)).contains($0) })

    let small = try await Engine(bundle: bundle, contextLimit: 8)
    await #expect(throws: EngineError.contextOverflow(needed: prefix.count, limit: 8)) {
        try await small.prefill(prefix)
    }
}
