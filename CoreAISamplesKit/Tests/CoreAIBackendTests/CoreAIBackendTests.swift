import Foundation
import FoundationModels
import Jinja
import Testing

@_spi(Verification) @testable import CoreAIBackend

let tinyMetadata = """
    {"metadata_version": "0.2", "kind": "llm", "name": "tiny",
     "assets": {"main": "tiny.aimodel"},
     "language": {"tokenizer": "x/tiny", "vocab_size": 151936, "max_context_length": 8192,
                  "embedded_tokenizer": true, "function_map": {"main": ["main"]}}}
    """

func tinyBundle(metadata: String = tinyMetadata) throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: folder.appending(path: "tiny.aimodel"), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: folder.appending(path: "tokenizer"), withIntermediateDirectories: true)
    try Data(metadata.utf8).write(to: folder.appending(path: "metadata.json"))
    try Data(#"{"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>"}"#.utf8)
        .write(to: folder.appending(path: "tokenizer/tokenizer_config.json"))
    try Data(#"""
        {"added_tokens": [{"id": 3, "content": "<|im_end|>", "special": true}],
         "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}}
        """#.utf8).write(to: folder.appending(path: "tokenizer/tokenizer.json"))
    return folder
}

@Test func readsMetadata() throws {
    let folder = try tinyBundle()
    defer { try? FileManager.default.removeItem(at: folder) }

    let bundle = try ModelBundle(url: folder)
    #expect(bundle.name == "tiny")
    #expect(bundle.modelURL == folder.appending(path: "tiny.aimodel"))
    #expect(bundle.tokenizerURL == folder.appending(path: "tokenizer"))
    #expect(bundle.vocabSize == 151936)
    #expect(bundle.maxContextLength == 8192)
}

@Test func bundleNamesWhatIsMissingOrWrong() throws {
    for (path, error) in [
        ("metadata.json", BundleError.missingFile("metadata.json")), ("tiny.aimodel", .missingFile("tiny.aimodel")),
        ("tokenizer", .missingFile("tokenizer/tokenizer.json")),
    ] {
        let folder = try tinyBundle()
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.removeItem(at: folder.appending(path: path))
        #expect(throws: error) { try ModelBundle(url: folder) }
    }

    for (metadata, error) in [
        (#"{"assets": {"main": "tiny.aimodel"}, "language": {}}"#, BundleError.missingField("name")),
        (#"{"name": "tiny", "assets": {}}"#, .missingField("assets.main")),
        (#"{"name": "tiny", "assets": {"main": "tiny.aimodel"}}"#, .missingField("language.vocab_size")),
        (#"{"name": "tiny", "assets": {"main": "tiny.aimodel"}, "language": {"vocab_size": 8}}"#, .missingField("language.max_context_length")),
        (#"{"name": "tiny", "assets": {"main": "tiny.aimodel"}, "language": {"vocab_size": 8, "max_context_length": 64, "function_map": {"main": []}}}"#,
         .missingField("language.function_map.main")),
        (#"{"name": "tiny", "assets": {"main": "tiny.aimodel"}, "language": {"vocab_size": 0, "max_context_length": 64, "function_map": {"main": ["main"]}}}"#,
         .invalidValue("language.vocab_size", 0)),
    ] {
        let folder = try tinyBundle(metadata: metadata)
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(throws: error) { try ModelBundle(url: folder) }
    }
    #expect(BundleError.missingField("language.vocab_size").localizedDescription == "metadata.json has no language.vocab_size.")

    for functionMap in ["", #", "function_map": {}"#, #", "function_map": {"main": ["main"]}"#] {
        let folder = try tinyBundle(metadata: #"{"name": "tiny", "assets": {"main": "tiny.aimodel"}, "language": {"vocab_size": 8, "max_context_length": 64"#
            + functionMap + "}}")
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(throws: Never.self) { try ModelBundle(url: folder) }
    }
}

@Test func readsTheLastDimensionOfAnOutputType() {
    #expect(Engine.lastDimension("NDArray (Float16, 1 \u{D7} ? \u{D7} 248320)") == 248320)
    #expect(Engine.lastDimension("NDArray (Float16, 1 \u{D7} ? \u{D7} 4096)") == 4096)
    #expect(Engine.lastDimension("NDArray (Float16, 3 \u{D7} 5)") == 5)
    #expect(Engine.lastDimension("NDArray (Float16, 7)") == 7)
    #expect(Engine.lastDimension("NDArray (Float16, 1 \u{D7} ? \u{D7} ?)") == nil)
    #expect(Engine.lastDimension("Image") == nil)
}

@Test func tokenizerMustFitTheVocabularySize() async throws {
    let fits = try tinyBundle(metadata: tinyMetadata.replacingOccurrences(of: "151936", with: "4"))
    let tooSmall = try tinyBundle(metadata: tinyMetadata.replacingOccurrences(of: "151936", with: "3"))
    defer {
        try? FileManager.default.removeItem(at: fits)
        try? FileManager.default.removeItem(at: tooSmall)
    }
    #expect(try ModelBundle.tokenCount(fits.appending(path: "tokenizer/tokenizer.json")) == 4)
    let tokenizer = try await ModelBundle(url: fits).loadTokenizer()
    #expect(tokenizer.eosTokenId == 3)
    await #expect(throws: BundleError.vocabularyMismatch(tokenizer: 4, metadata: 3)) {
        try await ModelBundle(url: tooSmall).loadTokenizer()
    }

    let unigram = fits.appending(path: "unigram.json")
    try Data(#"{"added_tokens": [], "model": {"type": "Unigram", "vocab": [["a", -1.0], ["b", -2.0]]}}"#.utf8).write(to: unigram)
    #expect(try ModelBundle.tokenCount(unigram) == 2)
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

@Test func reusePlanGoesBackToTheRightPoint() {
    typealias Stop = ReusePlan.Stop
    var plan = ReusePlan(processed: [1, 2, 3, 9, 9], saved: [], prompt: [1, 2, 3, 4, 5], hybrid: false, afterUser: 4) { 0 }
    #expect(plan.keep == 3)
    #expect(plan.stops == [Stop(position: 5, role: .generationStart)])

    plan = ReusePlan(processed: [1, 2, 3, 4, 5, 6], saved: [], prompt: [1, 2, 3], hybrid: false, afterUser: 2) { 0 }
    #expect(plan.keep == 2)
    #expect(plan.stops == [Stop(position: 3, role: .generationStart)])
    plan = ReusePlan(processed: [1, 2, 3, 4, 5, 6], saved: [3], prompt: [1, 2, 3], hybrid: false, afterUser: 2) { 0 }
    #expect(plan.keep == 3)
    #expect(plan.stops.isEmpty)

    plan = ReusePlan(processed: [1, 2, 3, 4, 5, 6, 7], saved: [2, 6], prompt: [1, 2, 3, 4, 9, 9, 9, 9], hybrid: true, afterUser: 6) { 0 }
    #expect(plan.keep == 2)
    #expect(plan.stops == [Stop(position: 6, role: .afterUser), Stop(position: 8, role: .generationStart)])

    let preamble = (0..<300).map(Int32.init)
    plan = ReusePlan(processed: preamble + [7, 7], saved: [302], prompt: preamble + [8, 9, 10], hybrid: true, afterUser: 302) { 300 }
    #expect(plan.keep == 0)
    #expect(plan.stops == [
        Stop(position: 300, role: .divergence), Stop(position: 302, role: .afterUser), Stop(position: 303, role: .generationStart),
    ])
    plan = ReusePlan(processed: preamble + [8, 9, 10, 11], saved: [300], prompt: preamble + [5, 6], hybrid: true, afterUser: 301) { 0 }
    #expect(plan.keep == 300)
    #expect(plan.stops == [Stop(position: 301, role: .afterUser), Stop(position: 302, role: .generationStart)])
    plan = ReusePlan(processed: Array(preamble[..<200]) + [7], saved: [], prompt: preamble, hybrid: true, afterUser: 299) { 0 }
    #expect(plan.keep == 0)
    #expect(plan.stops == [Stop(position: 299, role: .afterUser), Stop(position: 300, role: .generationStart)])

    plan = ReusePlan(processed: [1, 2, 3], saved: [], prompt: [1, 2, 3, 4, 5], hybrid: true, afterUser: 4) { 0 }
    #expect(plan.keep == 3)
    #expect(plan.stops == [Stop(position: 4, role: .afterUser), Stop(position: 5, role: .generationStart)])

    plan = ReusePlan(processed: [1, 2, 3, 4, 5, 6], saved: [3, 4], prompt: [1, 2, 3, 4], hybrid: true, afterUser: 3) { 0 }
    #expect(plan.keep == 4)
    #expect(plan.stops.isEmpty)
    plan = ReusePlan(processed: [1, 2, 3, 4, 5, 6], saved: [3], prompt: [1, 2, 3, 4], hybrid: true, afterUser: 3) { 0 }
    #expect(plan.keep == 3)
    #expect(plan.stops == [Stop(position: 4, role: .generationStart)])
}

@Test func reusePlanKeepsTheQuestionBoundaryThroughToolSteps() {
    typealias Stop = ReusePlan.Stop
    let question = Array<Int32>(1...6)
    let step1 = question + [7, 8]
    let generated = step1 + [20, 21, 22]
    var plan = ReusePlan(processed: generated, saved: [6, 8], prompt: generated + [30, 31, 32, 33], hybrid: true, afterUser: 6) { 0 }
    #expect(plan.keep == generated.count)
    #expect(plan.stops == [Stop(position: 15, role: .generationStart)])

    plan = ReusePlan(processed: generated, saved: [6, 8], prompt: step1 + [20, 29, 22, 30, 31], hybrid: true, afterUser: 6) { 0 }
    #expect(plan.shared == 9)
    #expect(plan.keep == 8)
    #expect(plan.stops == [Stop(position: 13, role: .generationStart)])

    let cancelled = generated + [40, 41, 42, 43, 50, 51]
    plan = ReusePlan(processed: cancelled, saved: [6, 15], prompt: generated + [44, 45, 46], hybrid: true, afterUser: 13) { 0 }
    #expect(plan.keep == 6)
    #expect(plan.stops == [Stop(position: 13, role: .afterUser), Stop(position: 14, role: .generationStart)])
}

@Test func reusePlanSavesTheDivergenceAtTheQuestionStart() {
    typealias Stop = ReusePlan.Stop
    let preamble = (0..<300).map(Int32.init)
    var plan = ReusePlan(processed: preamble + [5, 6, 7, 1, 1], saved: [305], prompt: preamble + [5, 6, 8, 2, 2, 2], hybrid: true, afterUser: 305) { 300 }
    #expect(plan.shared == 302)
    #expect(plan.keep == 0)
    #expect(plan.stops == [
        Stop(position: 300, role: .divergence), Stop(position: 305, role: .afterUser), Stop(position: 306, role: .generationStart),
    ])

    plan = ReusePlan(processed: preamble + [5, 6, 8, 2, 2, 2, 9], saved: [300, 305, 306], prompt: preamble + [5, 4, 4, 4, 4], hybrid: true, afterUser: 304) { 300 }
    #expect(plan.shared == 301)
    #expect(plan.keep == 300)
    #expect(plan.stops == [Stop(position: 304, role: .afterUser), Stop(position: 305, role: .generationStart)])

    plan = ReusePlan(processed: Array(preamble[..<280]) + [9, 9, 9], saved: [], prompt: preamble + [5, 6], hybrid: true, afterUser: 301) { 300 }
    #expect(plan.stops.first == Stop(position: 280, role: .divergence))

    let longer = (0..<500).map(Int32.init)
    plan = ReusePlan(processed: longer + [7], saved: [100], prompt: longer + [8, 8], hybrid: true, afterUser: 501) { 300 }
    #expect(plan.keep == 100)
    #expect(plan.stops == [Stop(position: 501, role: .afterUser), Stop(position: 502, role: .generationStart)])

    var rendered = 0
    let question = preamble + [5, 6]
    let steps = question + Array(repeating: 3, count: 400)
    plan = ReusePlan(processed: steps, saved: [302], prompt: question + [3, 3, 9, 9], hybrid: true, afterUser: 302) {
        rendered += 1
        return 300
    }
    #expect(plan.keep == 302)
    plan = ReusePlan(processed: steps, saved: [302], prompt: steps + [4, 4], hybrid: true, afterUser: 302) {
        rendered += 1
        return 300
    }
    #expect(plan.keep == steps.count)
    #expect(rendered == 0)
}

@Test func spliceRestoresTheGeneratedSplitWhereTheTextIsTheSame() {
    let pieces = [
        1: "<|im_start|>assistant\n", 2: "<|im_end|>\n<|im_start|>user\n<tool_response>\nsaved\n</tool_response><|im_end|>\n", 3: "Write a note.",
        4: "Write it.", 5: " ", 177636: "dingu", 5205: ",", 160305: "rei", 162532: "tore-", 161546: "shingu", 247202: "taiou",
        42420: "re", 75218: "ito", 149896: "re-",
    ]
    func decode(_ tokens: [Int]) -> String { tokens.map { pieces[$0] ?? "?" }.joined() }
    let question: [Int32] = [3, 1]
    let toolTurn: [Int32] = [2, 1]
    let generated: [Int32] = [177636, 5205, 160305, 162532, 161546, 247202]
    let rerendered: [Int32] = [177636, 5205, 42420, 75218, 149896, 161546, 247202]
    #expect(decode(generated.map(Int.init)) == decode(rerendered.map(Int.init)))

    var result = ReusePlan.splice(question + rerendered + toolTurn, processed: question + generated, generations: [(2, generated)], decode: decode)
    #expect(result.prompt == question + generated + toolTurn)
    #expect(result.spliced == 1)

    result = ReusePlan.splice(question + generated + toolTurn, processed: question + generated, generations: [(2, generated)], decode: decode)
    #expect(result.prompt == question + generated + toolTurn)
    #expect(result.spliced == 0)

    let spaced = question + [177636, 5205, 5, 42420, 75218, 149896, 161546, 247202] + toolTurn
    result = ReusePlan.splice(spaced, processed: question + generated, generations: [(2, generated)], decode: decode)
    #expect(result.prompt == spaced)
    #expect(result.spliced == 0)

    result = ReusePlan.splice(question + rerendered + toolTurn, processed: [4, 1] + generated, generations: [(2, generated)], decode: decode)
    #expect(result.prompt == question + rerendered + toolTurn)
    #expect(result.spliced == 0)

    let first: [Int32] = [160305, 162532, 161546]
    let processed = question + first + toolTurn + generated
    let prompt = question + [42420, 75218, 149896, 161546] + toolTurn + rerendered + toolTurn
    result = ReusePlan.splice(prompt, processed: processed, generations: [(7, generated), (2, first)], decode: decode)
    #expect(result.prompt == processed + toolTurn)
    #expect(result.spliced == 2)
    result = ReusePlan.splice(prompt, processed: processed, generations: [(7, generated)], decode: decode)
    #expect(result.prompt == prompt)
    #expect(result.spliced == 0)
}

@Test func toolOutputCutsDropTheOldestGroupsAndHalveTheLast() {
    let groups: [[(id: String, count: Int)]] = [[("a", 3000)], [("b", 3000), ("c", 800)], [("d", 3000)], [("e", 1800), ("f", 700), ("g", 300)]]
    var cuts = ToolOutputCut.shrink(groups, [:])
    #expect(cuts == ["a": .dropped] as [String: ToolOutputCut])
    cuts = ToolOutputCut.shrink(groups, cuts ?? [:])
    #expect(cuts == ["a": .dropped, "b": .dropped, "c": .dropped])
    cuts = ToolOutputCut.shrink(groups, cuts ?? [:])
    #expect(cuts == ["a": .dropped, "b": .dropped, "c": .dropped, "e": .prefix(900), "f": .prefix(500)])
    cuts = ToolOutputCut.shrink(groups, cuts ?? [:])
    #expect(cuts == ["a": .dropped, "b": .dropped, "c": .dropped, "e": .prefix(500), "f": .prefix(500)])
    #expect(ToolOutputCut.shrink(groups, cuts ?? [:]) == nil)

    #expect(ToolOutputCut.shrink([[("x", 4000)], [("y", 600)]], [:]) == ["y": .prefix(500)])
    #expect(ToolOutputCut.shrink([[("x", 4000)], [("y", 500)]], [:]) == nil)
    #expect(ToolOutputCut.shrink([], [:]) == nil)
    #expect(ToolOutputCut.prefix(3).apply("abcdef") == "abc")
    #expect(ToolOutputCut.dropped.apply("abcdef") == "[truncated]")
}

@Test func fitShortensToolOutputsAndRemembersByEntry() throws {
    typealias Cuts = [String: ToolOutputCut]
    func text(_ content: String) -> [Transcript.Segment] { [.text(.init(content: content))] }
    func call(_ id: String) throws -> Transcript.Entry {
        .toolCalls(.init([Transcript.ToolCall(id: id, toolName: "read_page", arguments: try GeneratedContent(json: #"{"name": "x"}"#))]))
    }
    func page(_ id: String, _ count: Int) -> Transcript.Entry {
        .toolOutput(.init(id: id, toolName: "read_page", segments: text(String(repeating: "p", count: count))))
    }
    let characters: ([[String: any Sendable]]) -> [Int32] = { messages in
        Array(repeating: 0, count: messages.reduce(0) { $0 + ($1["content"] as? String ?? "").count })
    }
    func contents(_ messages: [[String: any Sendable]]) -> [Int] {
        messages.filter { $0["role"] as? String == "tool" }.map { ($0["content"] as? String ?? "").count }
    }
    var entries: [Transcript.Entry] = [.prompt(.init(segments: text("Read three pages.")))]
    entries += [try call("c1"), page("o1", 2000), try call("c2"), page("o2", 2000), try call("c3"), page("o3", 2000)]
    #expect(entries[2].id == "o1")

    var fitted = try Runtime.fit(Transcript(entries: entries), remembered: [:], limit: 5000, reserve: 1000, render: characters)
    var expected: Cuts = ["o1": .dropped, "o3": .prefix(1000)]
    #expect(fitted.cuts == expected)
    #expect(contents(fitted.messages) == [11, 2000, 1000])
    #expect(fitted.prompt.count == 3028)

    entries += [try call("c4"), page("o4", 500)]
    let remembered = fitted.cuts.merging(["gone": ToolOutputCut.dropped]) { a, _ in a }
    fitted = try Runtime.fit(Transcript(entries: entries), remembered: remembered, limit: 5000, reserve: 1000, render: characters)
    #expect(fitted.cuts == expected)
    #expect(contents(fitted.messages) == [11, 2000, 1000, 500])

    entries += [try call("c5"), page("o5", 4000)]
    fitted = try Runtime.fit(Transcript(entries: entries), remembered: fitted.cuts, limit: 5000, reserve: 1000, render: characters)
    expected = ["o1": .dropped, "o2": .dropped, "o3": .dropped, "o5": .prefix(2000)]
    #expect(fitted.cuts == expected)
    #expect(contents(fitted.messages) == [11, 11, 11, 500, 2000])

    let plain = try Runtime.fit(Transcript(entries: entries), remembered: [:], limit: 20000, reserve: nil, render: characters)
    #expect(plain.cuts.isEmpty)
    #expect(plain.prompt.count == 10517)
}

@Test func fitGoesOnInWhatIsLeftAndFailsOnlyWhenThePromptDoesNotFit() throws {
    func text(_ content: String) -> [Transcript.Segment] { [.text(.init(content: content))] }
    func call(_ id: String) throws -> Transcript.Entry {
        .toolCalls(.init([Transcript.ToolCall(id: id, toolName: "read_page", arguments: try GeneratedContent(json: #"{"name": "x"}"#))]))
    }
    func page(_ id: String, _ count: Int) -> Transcript.Entry {
        .toolOutput(.init(id: id, toolName: "read_page", segments: text(String(repeating: "p", count: count))))
    }
    let characters: ([[String: any Sendable]]) -> [Int32] = { messages in
        Array(repeating: 0, count: messages.reduce(0) { $0 + ($1["content"] as? String ?? "").count })
    }
    let settled: [Transcript.Entry] = [
        .prompt(.init(segments: text(String(repeating: "q", count: 4000)))), try call("c1"), page("o1", 300), try call("c2"), page("o2", 300),
    ]
    let fitted = try Runtime.fit(Transcript(entries: settled), remembered: [:], limit: 5000, reserve: 1000, render: characters)
    #expect(fitted.cuts.isEmpty)
    #expect(fitted.prompt.count == 4600)

    let long = Transcript(entries: [.prompt(.init(segments: text(String(repeating: "q", count: 5000))))])
    for reserve in [1000, nil] {
        #expect {
            try Runtime.fit(long, remembered: [:], limit: 5000, reserve: reserve, render: characters)
        } throws: { error in
            guard case LanguageModelError.contextSizeExceeded(let exceeded) = error else { return false }
            return exceeded.contextSize == 5000 && exceeded.tokenCount == 5000
        }
    }
}

@Test func cutMemoryStaysWithEachConversation() throws {
    func text(_ content: String) -> [Transcript.Segment] { [.text(.init(content: content))] }
    func call(_ id: String) throws -> Transcript.Entry {
        .toolCalls(.init([Transcript.ToolCall(id: id, toolName: "read_page", arguments: try GeneratedContent(json: #"{"name": "x"}"#))]))
    }
    func page(_ id: String, _ count: Int) -> Transcript.Entry {
        .toolOutput(.init(id: id, toolName: "read_page", segments: text(String(repeating: "p", count: count))))
    }
    let characters: ([[String: any Sendable]]) -> [Int32] = { messages in
        Array(repeating: 0, count: messages.reduce(0) { $0 + ($1["content"] as? String ?? "").count })
    }
    func conversation(_ name: String) throws -> [Transcript.Entry] {
        [.prompt(.init(segments: text("Read three pages.")))] + (try (1...3).flatMap { [try call("\(name)c\($0)"), page("\(name)o\($0)", 2000)] })
    }
    var memory: [String: ToolOutputCut] = [:]
    func step(_ entries: [Transcript.Entry]) throws -> (cuts: [String: ToolOutputCut], truncated: Int) {
        let fitted = try Runtime.fit(Transcript(entries: entries), remembered: memory, limit: 5000, reserve: 1000, render: characters)
        return (fitted.cuts, Runtime.remember(fitted.cuts, in: &memory))
    }

    var a = try conversation("a")
    let a1 = try step(a)
    #expect(a1.cuts == ["ao1": .dropped, "ao3": .prefix(1000)])
    #expect(a1.truncated == 2)
    let b1 = try step(try conversation("b"))
    #expect(b1.cuts == ["bo1": .dropped, "bo3": .prefix(1000)])
    #expect(b1.truncated == 2)
    a += [try call("ac4"), page("ao4", 500)]
    let a2 = try step(a)
    #expect(a2.cuts == a1.cuts)
    #expect(a2.truncated == 0)
    #expect(memory.count == 4)

    var full = Dictionary(uniqueKeysWithValues: (0..<Runtime.rememberedCalls).map { ("old\($0)", ToolOutputCut.dropped) })
    #expect(Runtime.remember(["new": .dropped], in: &full) == 1)
    #expect(full == ["new": .dropped])
}

@Generable struct AddArguments {
    var a: Int
    var b: Int
}

@Test func requiredToolCallingIsRefused() throws {
    let add = Transcript.ToolDefinition(name: "add", description: "Add two integers.", parameters: AddArguments.generationSchema)
    func request(_ mode: GenerationOptions.ToolCallingMode?) -> LanguageModelExecutorGenerationRequest {
        LanguageModelExecutorGenerationRequest(
            id: UUID(), transcript: Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "Add 1 and 2."))]))]),
            enabledTools: [add], generationOptions: GenerationOptions(toolCallingMode: mode), contextOptions: ContextOptions(), metadata: [:])
    }
    #expect(try Runtime.tools(request(nil)).map(\.name) == ["add"])
    #expect(try Runtime.tools(request(.allowed)).map(\.name) == ["add"])
    #expect(try Runtime.tools(request(.disallowed)).isEmpty)
    #expect {
        try Runtime.tools(request(.required))
    } throws: { error in
        guard case LanguageModelError.unsupportedCapability(let unsupported) = error else { return false }
        return unsupported.capability == .guidedGeneration && unsupported.debugDescription.hasSuffix("Use .allowed.")
    }
}

@Test func toolCallParserReadsBothFormats() {
    let types = ["add": ["a": "integer", "b": "integer"], "get_weather": ["city": "string"]]
    func parse(_ chunks: [String]) -> [ToolCallParser.Event] {
        var parser = ToolCallParser(types: types)
        return chunks.flatMap { parser.feed($0) } + parser.finish()
    }

    #expect(parse(["<tool_call>", "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n", "</tool_call>"])
        == [.call(name: "get_weather", arguments: #"{"city":"Tokyo"}"#)])
    #expect(parse([#"<tool_call>{"name": "add", "arguments": "{\"b\": 2, \"a\": 1}"}</tool_call>"#])
        == [.call(name: "add", arguments: #"{"b":2,"a":1}"#)])
    #expect(parse([#"<tool_call>{"name": "add", "arguments": {"b": 2, "a": 1}}</tool_call>"#])
        == [.call(name: "add", arguments: #"{"b":2,"a":1}"#)])
    #expect(parse([#"<tool_call>{"arguments": {"z": "a, \"b\": {c}\\", "y": [1, {"x": ":"}], "w": null}, "name": "add"}</tool_call>"#])
        == [.call(name: "add", arguments: #"{"z":"a, \"b\": {c}\\","y":[1, {"x": ":"}],"w":null}"#)])
    #expect(parse([#"<tool_call>{"name": "add", "arguments": 5}</tool_call>"#]) == [.text(#"<tool_call>{"name": "add", "arguments": 5}</tool_call>"#)])
    #expect(parse(["<tool_call>\n<function=add>\n<parameter=b>\n25\n</parameter>\n<parameter=a>\n17\n</parameter>\n</function>\n</tool_call>"])
        == [.call(name: "add", arguments: #"{"b":25,"a":17}"#)])
    #expect(parse(["<tool_call>\n<function=get_weather>\n<parameter=city>\n42\n</parameter>\n<parameter=note>\ntrue\n</parameter>\n</function>\n</tool_call>"])
        == [.call(name: "get_weather", arguments: #"{"city":"42","note":true}"#)])
    #expect(parse(["<tool_call>\n<function=add>\n<parameter=a>\nseventeen\n</parameter>\n<parameter=b>\n[1, 2]\n</parameter>\n</function>\n</tool_call>"])
        == [.call(name: "add", arguments: #"{"a":"seventeen","b":[1, 2]}"#)])

    let weather = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    let sum = "<tool_call>\n<function=add>\n<parameter=a>\n17\n</parameter>\n<parameter=b>\n25\n</parameter>\n</function>\n</tool_call>"
    #expect(parse(["Let me check.\n\n", weather, "\n", sum])
        == [.text("Let me check."), .call(name: "get_weather", arguments: #"{"city":"Paris"}"#), .call(name: "add", arguments: #"{"a":17,"b":25}"#)])
    #expect(parse(["<tool", "_call>\n<function=get", "_weather>\n<parameter=city>\nTo", "kyo\n</parameter>\n</function>\n</tool", "_call>"])
        == [.call(name: "get_weather", arguments: #"{"city":"Tokyo"}"#)])

    var parser = ToolCallParser(types: types)
    #expect(parser.feed("Paris is") == [.text("Paris is")])
    #expect(parser.feed(" sunny.\n<") == [.text(" sunny.")])
    #expect(parser.feed("b>") == [.text("\n<b>")])
    #expect(parser.finish().isEmpty)

    #expect(parse(["<tool_call>\n<function=get_weather>\n<param"]) == [.text("<tool_call>\n<function=get_weather>\n<param")])
    #expect(parse(["Hi\n", "<tool_call>\n<function=>\n</function>\n</tool_call>"]) == [.text("Hi"), .text("\n<tool_call>\n<function=>\n</function>\n</tool_call>")])
    #expect(parse([#"<tool_call>{"name": "add", "arguments": {"a": }}</tool_call>"#]) == [.text(#"<tool_call>{"name": "add", "arguments": {"a": }}</tool_call>"#)])
    #expect(parse([#"<tool_call>{"arguments": {"a": 1}}</tool_call>"#]) == [.text(#"<tool_call>{"arguments": {"a": 1}}</tool_call>"#)])
    #expect(parse(["<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n</tool_call>"]) == [.text("<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n</tool_call>")])
}

@Test func transcriptRebuildsToolCallsAndOutputs() throws {
    func text(_ content: String) -> [Transcript.Segment] { [.text(.init(content: content))] }
    let add = Transcript.ToolDefinition(name: "add", description: "Add two integers.", parameters: AddArguments.generationSchema)
    let schema = ToolSchema(add)
    #expect(schema.order == ["a", "b"])
    #expect(schema.types == ["a": "integer", "b": "integer"])
    #expect(try JSON.dumps(schema.spec) == #"{"type": "function", "function": {"name": "add", "description": "Add two integers.", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}"#)

    let transcript = Transcript(entries: [
        .instructions(.init(segments: [], toolDefinitions: [add])),
        .prompt(.init(segments: text("What is 1 plus 2?"))),
        .reasoning(.init(segments: text("Use the tool."))),
        .response(.init(segments: text("Let me add."))),
        .toolCalls(.init([
            Transcript.ToolCall(id: "c1", toolName: "add", arguments: try GeneratedContent(json: #"{"b": 2, "a": 1}"#)),
            Transcript.ToolCall(id: "c2", toolName: "add", arguments: try GeneratedContent(json: #"{"a": 3, "b": 4}"#)),
        ])),
        .toolOutput(.init(id: "c1", toolName: "add", segments: text("3"))),
        .toolOutput(.init(id: "c2", toolName: "add", segments: text("7"))),
        .reasoning(.init(segments: text("Done."))),
        .response(.init(segments: text("1 plus 2 is 3."))),
        .prompt(.init(segments: text("Thanks."))),
    ])
    let messages = try Runtime.messages(transcript)
    #expect(messages.map { $0["role"] as? String } == ["user", "assistant", "tool", "tool", "assistant", "user"])
    #expect(messages.map { $0["content"] as? String } == ["What is 1 plus 2?", "Let me add.", "3", "7", "1 plus 2 is 3.", "Thanks."])
    #expect(messages[1]["reasoning_content"] as? String == "Use the tool.")
    #expect(messages[4]["reasoning_content"] as? String == "Done.")
    let calls = try #require(messages[1]["tool_calls"] as? [Value])
    #expect(try calls.map { try JSON.dumps($0) } == [
        #"{"type": "function", "function": {"name": "add", "arguments": {"a": 1, "b": 2}}}"#,
        #"{"type": "function", "function": {"name": "add", "arguments": {"a": 3, "b": 4}}}"#,
    ])
    #expect(try JSON.dumps(Runtime.value(GeneratedContent(json: #"{"x": 2.5, "n": 17, "s": "a", "f": true, "z": null, "l": [1]}"#), order: ["n", "x"]))
        .hasPrefix(#"{"n": 17, "x": 2.5, "#))
}

@Test func historyDrawsToolCallArgumentsInTheWrittenOrder() throws {
    let add = Transcript.ToolDefinition(name: "add", description: "Add two integers.", parameters: AddArguments.generationSchema)
    let written = [#"{"b":2,"a":1}"#, #"{"b":4,"a":3}"#].map { ToolCallParser.members($0)?.map(\.key) }
    #expect(written == [["b", "a"], ["b", "a"]])
    #expect(ToolCallParser.members("{}")?.isEmpty == true)
    #expect(ToolCallParser.members(#"{"a": }"#) == nil)

    let transcript = Transcript(entries: [
        .instructions(.init(segments: [], toolDefinitions: [add])),
        .prompt(.init(segments: [.text(.init(content: "Add."))])),
        .toolCalls(.init([
            Transcript.ToolCall(id: "c1", toolName: "add", arguments: try GeneratedContent(json: #"{"a": 1, "b": 2}"#)),
            Transcript.ToolCall(id: "c2", toolName: "add", arguments: try GeneratedContent(json: #"{"a": 3, "b": 4}"#)),
            Transcript.ToolCall(id: "c3", toolName: "add", arguments: try GeneratedContent(json: #"{"a": 5, "b": 6}"#)),
            Transcript.ToolCall(id: "c4", toolName: "add", arguments: try GeneratedContent(json: #"{"b": 8, "a": 7}"#)),
        ])),
    ])
    let fitted = try Runtime.fit(
        transcript, remembered: [:], written: ["c1": written[0] ?? [], "c2": written[1] ?? [], "c3": ["b", "b"]], limit: 10, reserve: nil
    ) { _ in [0] }
    let calls = try #require(fitted.messages[1]["tool_calls"] as? [Value])
    #expect(try calls.map { try JSON.dumps($0) } == [
        #"{"type": "function", "function": {"name": "add", "arguments": {"b": 2, "a": 1}}}"#,
        #"{"type": "function", "function": {"name": "add", "arguments": {"b": 4, "a": 3}}}"#,
        #"{"type": "function", "function": {"name": "add", "arguments": {"b": 6, "a": 5}}}"#,
        #"{"type": "function", "function": {"name": "add", "arguments": {"a": 7, "b": 8}}}"#,
    ])
}

func merged(_ parts: [ThinkParser.Part]) -> [ThinkParser.Part] {
    parts.reduce(into: []) { result, part in
        switch (result.last, part) {
        case (.reasoning(let a)?, .reasoning(let b)): result[result.count - 1] = .reasoning(a + b)
        case (.response(let a)?, .response(let b)): result[result.count - 1] = .response(a + b)
        default: result.append(part)
        }
    }
}

@Test func thinkParserSeparatesReasoningFromTheAnswer() {
    var parser = ThinkParser(reasoning: false)
    var parts: [ThinkParser.Part] = []
    for chunk in ["<th", "ink>", "\n", "Let me", " think.", "\n</", "think>", "\n\n", "Paris", "."] {
        parts += parser.feed(chunk)
    }
    #expect(merged(parts + parser.finish()) == [.reasoning("Let me think.\n"), .response("Paris.")])

    parser = ThinkParser(reasoning: true)
    #expect(merged(parser.feed("Hmm") + parser.feed("</think>\n\nOK") + parser.finish()) == [.reasoning("Hmm"), .response("OK")])

    parser = ThinkParser(reasoning: false)
    #expect(parser.feed("Paris") == [.response("Paris")])
    #expect(parser.feed(" is <thi") == [.response(" is ")])
    #expect(parser.finish() == [.response("<thi")])

    #expect(ThinkParser.opensReasoning("<|im_start|>assistant\n<think>\n"))
    #expect(!ThinkParser.opensReasoning("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    #expect(!ThinkParser.opensReasoning("<|im_start|>assistant\n"))

    let pieces = [1: "<think>", 2: "x", 3: "</think>", 4: "<|im_start|>assistant\n"]
    func decode(_ tokens: [Int]) -> String { tokens.map { pieces[$0] ?? "" }.joined() }
    let filler = Array<Int32>(repeating: 2, count: 15)
    #expect(!ThinkParser.opensReasoning(after: [1] + filler + [4], decode: decode))
    #expect(ThinkParser.opensReasoning(after: [1] + filler.dropLast() + [4], decode: decode))
    #expect(ThinkParser.opensReasoning(after: filler + filler + [4, 1, 2], decode: decode))
    #expect(!ThinkParser.opensReasoning(after: filler + [4, 1, 2, 3, 2], decode: decode))
}

@Test func transcriptKeepsReasoningForTheTemplate() throws {
    func text(_ content: String) -> [Transcript.Segment] { [.text(.init(content: content))] }
    let transcript = Transcript(entries: [
        .instructions(.init(segments: text("Be brief."), toolDefinitions: [])),
        .prompt(.init(segments: text("Hi"))),
        .reasoning(.init(segments: text("The user greets."))),
        .response(.init(segments: text("Hello."))),
        .prompt(.init(segments: text("Bye"))),
    ])
    let messages = try Runtime.messages(transcript)
    #expect(messages.map { $0["role"] as? String } == ["system", "user", "assistant", "user"])
    #expect(messages.map { $0["content"] as? String } == ["Be brief.", "Hi", "Hello.", "Bye"])
    #expect(messages[2]["reasoning_content"] as? String == "The user greets.")
    #expect(messages[1]["reasoning_content"] == nil)

    #expect(Runtime.templateContext(nil).isEmpty)
    #expect(Runtime.templateContext(.light)["reasoning_effort"] as? String == "low")
    #expect(Runtime.templateContext(.moderate)["reasoning_effort"] as? String == "medium")
    #expect(Runtime.templateContext(.deep)["reasoning_effort"] as? String == "xhigh")
    #expect(Runtime.templateContext(.custom("off"))["enable_thinking"] as? Bool == false)
}

@Test func samplerHonorsModeTemperatureAndSeed() {
    let logits: [Float16] = [1, 4, 3, 0, 2]
    func draws(_ mode: Sampler.Mode, temperature: Float = 1, seed: UInt64?, count: Int = 400) -> [Int32] {
        var sampler = Sampler(mode: mode, temperature: temperature, seed: seed)
        return (0..<count).map { _ in logits.withUnsafeBufferPointer { sampler.next($0) } }
    }
    #expect(Set(draws(.greedy, seed: nil)) == [1])
    #expect(Set(draws(.topK(5), temperature: 0, seed: 1)) == [1])
    #expect(Set(draws(.topK(1), seed: 1)) == [1])
    #expect(Set(draws(.topK(2), seed: 1)) == [1, 2])
    #expect(Set(draws(.topP(0.5), seed: 1)) == [1])
    #expect(Set(draws(.topP(0.8), seed: 1)) == [1, 2])
    #expect(Set(draws(.full, seed: 1, count: 4000)) == [0, 1, 2, 3, 4])
    #expect(draws(.topK(3), seed: 7) == draws(.topK(3), seed: 7))
    #expect(draws(.topK(3), seed: 7) != draws(.topK(3), seed: 8))
    #expect(draws(.topP(0.95), seed: 7) == draws(.topP(0.95), seed: 7))
    #expect(draws(.topP(0.95), seed: 7) != draws(.topP(0.95), seed: 8))
}

@Test func reuseOffReadsTheWholePromptFromAReset() {
    typealias Stop = ReusePlan.Stop
    let processed = (0..<300).map(Int32.init) + [5, 6, 7, 8]
    let prompt = processed + [9, 9]
    for hybrid in [true, false] {
        let on = Runtime.plan(prompt, processed: processed, saved: [302], hybrid: hybrid, reusing: true, afterUser: 304) { 300 }
        #expect(on == ReusePlan(processed: processed, saved: [302], prompt: prompt, hybrid: hybrid, afterUser: 304) { 300 })
        #expect(on.keep == processed.count)
        let off = Runtime.plan(prompt, processed: processed, saved: [302], hybrid: hybrid, reusing: false, afterUser: 304) { 300 }
        #expect(off.keep == 0)
        #expect(off.stops == (hybrid ? [Stop(position: 304, role: .afterUser)] : []) + [Stop(position: prompt.count, role: .generationStart)])
    }
}

@Test func droppingHistoryReasoningKeepsTheRestOfTheHistory() throws {
    func text(_ content: String) -> [Transcript.Segment] { [.text(.init(content: content))] }
    func call(_ id: String, _ json: String) throws -> Transcript.Entry {
        .toolCalls(.init([Transcript.ToolCall(id: id, toolName: "add", arguments: try GeneratedContent(json: json))]))
    }
    let add = Transcript.ToolDefinition(name: "add", description: "Add two integers.", parameters: AddArguments.generationSchema)
    let transcript = Transcript(entries: [
        .instructions(.init(segments: text("Be brief."), toolDefinitions: [add])),
        .prompt(.init(segments: text("What is 1 plus 2?"))),
        .reasoning(.init(segments: text("Use the tool."))),
        try call("c1", #"{"a": 1, "b": 2}"#),
        .toolOutput(.init(id: "c1", toolName: "add", segments: text("3"))),
        .reasoning(.init(segments: text("Done."))),
        .response(.init(segments: text("3."))),
        .prompt(.init(segments: text("And 2 plus 2?"))),
        .reasoning(.init(segments: text("Add again."))),
        try call("c2", #"{"a": 2, "b": 2}"#),
        .toolOutput(.init(id: "c2", toolName: "add", segments: text("4"))),
    ])
    func outline(_ messages: [[String: any Sendable]]) throws -> [String] {
        try messages.map { message in
            let calls = try (message["tool_calls"] as? [Value] ?? []).map { try JSON.dumps($0) }
            return ([message["role"] as? String ?? "", message["content"] as? String ?? ""] + calls).joined(separator: "|")
        }
    }
    let kept = try Runtime.messages(transcript)
    let dropped = try Runtime.messages(transcript, keepsReasoning: false)
    #expect(kept.compactMap { $0["reasoning_content"] as? String } == ["Use the tool.", "Done.", "Add again."])
    #expect(dropped.allSatisfy { $0["reasoning_content"] == nil })
    #expect(try outline(dropped) == outline(kept))
    #expect(try outline(kept).count == 8)

    let fitted = try Runtime.fit(transcript, remembered: [:], keepsReasoning: false, limit: 10, reserve: nil) { _ in [0] }
    #expect(fitted.messages.allSatisfy { $0["reasoning_content"] == nil })
    let fittedDefault = try Runtime.fit(transcript, remembered: [:], limit: 10, reserve: nil) { _ in [0] }
    #expect(fittedDefault.messages.compactMap { $0["reasoning_content"] as? String }.count == 3)
}

@Test func templateVariablesReachTheRendering() async throws {
    #expect(Runtime.templateContext(.light, variables: ["preserve_thinking": false])["reasoning_effort"] as? String == "low")
    #expect(Runtime.templateContext(.light, variables: ["preserve_thinking": false])["preserve_thinking"] as? Bool == false)
    #expect(Runtime.templateContext(.custom("off"), variables: ["enable_thinking": true])["enable_thinking"] as? Bool == true)
    #expect(Runtime.templateContext(nil, variables: [:]).isEmpty)

    let folder = try tinyBundle()
    defer { try? FileManager.default.removeItem(at: folder) }
    let tokenizer = try await ModelBundle(url: folder).loadTokenizer()
    let template = "{%- if preserve_thinking is undefined or preserve_thinking is true %}b{%- else %}a{%- endif %}"
    func render(_ variables: [String: Bool]) throws -> [Int] {
        let context = Runtime.templateContext(nil, variables: variables)
        return try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "x"]], chatTemplate: .literal(template), addGenerationPrompt: false, truncation: false,
            maxLength: nil, tools: nil, additionalContext: context.isEmpty ? nil : context)
    }
    #expect(try render([:]) == [1])
    #expect(try render(["preserve_thinking": true]) == [1])
    #expect(try render(["preserve_thinking": false]) == [0])
}

@Test func stateCheckPassesTheSameTopAndNearTiesOnly() {
    func up(_ value: Float16, _ ticks: Int) -> Float16 { (0..<ticks).reduce(value) { next, _ in next.nextUp } }
    func pair(_ high: Int, _ low: Int, ticks: Int) -> [Float16] {
        var values: [Float16] = [1, 0, 0, -2, 3]
        values[low] = 20
        values[high] = up(20, ticks)
        return values
    }
    #expect(StateCheck.ticks(up(16, 3), 16) == 3)
    #expect(StateCheck.ticks(20, 19) == 64)
    #expect(StateCheck.ticks(-0.0, 0) == 0)
    #expect(StateCheck.ticks(Float16.leastNonzeroMagnitude, -Float16.leastNonzeroMagnitude) == 2)
    #expect(StateCheck.top2([5, 5, 1]) == (0, 1))
    #expect(StateCheck.top2([1, 5, 7, 6]) == (2, 3))

    let base: [Float16] = [1, 20, 3, -2, 19]
    var check = StateCheck(current: base, reread: base)
    #expect(check.passed && check.currentTop == 1 && check.rereadTop == 1 && check.marginTicks == 64 && check.maxAbsDiff == 0)
    var shifted = base
    shifted[3] = 2
    check = StateCheck(current: base, reread: shifted)
    #expect(check.passed && check.maxAbsDiff == 4)

    for ticks in [1, 8] {
        check = StateCheck(current: pair(1, 2, ticks: ticks), reread: pair(2, 1, ticks: ticks))
        #expect(check.passed && check.currentTop == 1 && check.rereadTop == 2 && check.marginTicks == ticks)
        #expect(check.maxAbsDiff == Float(ticks) / 64)
    }
    check = StateCheck(current: pair(1, 2, ticks: 9), reread: pair(2, 1, ticks: 9))
    #expect(!check.passed && check.marginTicks == 9)
    check = StateCheck(current: pair(1, 2, ticks: 2), reread: pair(2, 4, ticks: 2))
    #expect(!check.passed && check.marginTicks == 2)
}

let bundlePaths = (ProcessInfo.processInfo.environment["COREAI_TEST_BUNDLES"] ?? "")
    .split(separator: ":").map(String.init)

let checkPaths = (ProcessInfo.processInfo.environment["COREAI_CHECK_BUNDLES"] ?? "")
    .split(separator: ":").map(String.init)

@Test(.enabled(if: !checkPaths.isEmpty), arguments: checkPaths)
func bundleOpensWithoutLoadingTheModel(path: String) async throws {
    let bundle = try ModelBundle(url: URL(filePath: path))
    let tokenizer = try await bundle.loadTokenizer()
    #expect(tokenizer.eosTokenId.map { $0 < bundle.vocabSize } == true)
}

let nonLanguageModelPaths = (ProcessInfo.processInfo.environment["COREAI_NONLM_BUNDLES"] ?? "")
    .split(separator: ":").map(String.init)

@Test(.enabled(if: !nonLanguageModelPaths.isEmpty), arguments: nonLanguageModelPaths)
func engineRefusesABundleThatIsNotALanguageModel(path: String) async throws {
    let bundle = try ModelBundle(url: URL(filePath: path))
    let width = try #require(Engine.mainOutputWidth(bundle.modelURL))
    #expect(width != bundle.vocabSize)
    await #expect(throws: BundleError.notLanguageModel(outputWidth: width, vocabSize: bundle.vocabSize)) {
        try await LocalLanguageModel(bundle: URL(filePath: path))
    }
}

@Test(.enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
func modelRunsAndRestores(path: String) async throws {
    let bundle = try ModelBundle(url: URL(filePath: path))
    #expect(Engine.mainOutputWidth(bundle.modelURL) == bundle.vocabSize)
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

func texts(_ session: LanguageModelSession) -> String {
    session.transcript.map { entry -> String in
        let (kind, segments): (String, [Transcript.Segment]) =
            switch entry {
            case .prompt(let prompt): ("prompt", prompt.segments)
            case .reasoning(let reasoning): ("reasoning", reasoning.segments)
            case .response(let response): ("response", response.segments)
            default: ("other", [])
            }
        return kind + ": " + segments.map { if case .text(let text) = $0 { text.content } else { "" } }.joined()
    }.joined(separator: "\n")
}

@Test(.enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
func sessionReusesStopsSamplesAndCancels(path: String) async throws {
    let model = try await LocalLanguageModel(bundle: URL(filePath: path))
    func stopReason(_ session: LanguageModelSession) -> String? {
        guard case .response(let response) = session.transcript.last else { return nil }
        return try? response.metadata["stopReason"]?.value(String.self)
    }
    func prefilled(_ session: LanguageModelSession) -> Int? {
        guard case .response(let response) = session.transcript.last else { return nil }
        return try? response.metadata["prefillTokens"]?.value(Int.self)
    }

    let session = LanguageModelSession(model: model, instructions: "You are a concise assistant.")
    for (turn, prompt) in ["My name is Taro. Remember it.", "What is the capital of France? Answer with one word.", "What is my name?"].enumerated() {
        let response = try await session.respond(to: prompt, options: GenerationOptions(maximumResponseTokens: 512))
        let usage = response.usage
        #expect(stopReason(session) == "stopToken")
        #expect(usage.input.cachedTokenCount + (prefilled(session) ?? -1) == usage.input.totalTokenCount)
        #expect((usage.input.cachedTokenCount > 0) == (turn > 0))
        #expect(await model.runtime.processed.count == usage.input.totalTokenCount + usage.output.totalTokenCount)
    }

    let direct = try await LanguageModelSession(model: model).respond(
        to: "What is the capital of France? Answer with one word.", options: GenerationOptions(maximumResponseTokens: 64),
        contextOptions: ContextOptions(reasoningLevel: .custom("off")))
    #expect(direct.usage.output.reasoningTokenCount == 0)
    #expect(direct.content.contains("Paris"))

    let limited = LanguageModelSession(model: model)
    let short = try await limited.respond(to: "Explain why the sky is blue.", options: GenerationOptions(maximumResponseTokens: 8))
    #expect(stopReason(limited) == "maxTokens")
    #expect(short.usage.output.totalTokenCount == 8)
    #expect(await model.runtime.processed.count == short.usage.input.totalTokenCount + 7)

    func sample(_ mode: GenerationOptions.SamplingMode) async throws -> String {
        let options = GenerationOptions(samplingMode: mode, maximumResponseTokens: 24)
        let session = LanguageModelSession(model: model)
        try await session.respond(to: "Write a short poem about the sea.", options: options)
        return texts(session)
    }
    #expect(try await sample(.greedy) == sample(.greedy))
    #expect(try await sample(.random(top: 40, seed: 7)) == sample(.random(top: 40, seed: 7)))
    #expect(try await sample(.random(top: 40, seed: 7)) != sample(.random(top: 40, seed: 8)))
    #expect(try await sample(.random(probabilityThreshold: 0.95, seed: 7)) == sample(.random(probabilityThreshold: 0.95, seed: 7)))
    #expect(try await sample(.random(probabilityThreshold: 0.95, seed: 7)) != sample(.random(probabilityThreshold: 0.95, seed: 8)))

    let cancelled = LanguageModelSession(model: model)
    let task = Task { try await cancelled.respond(to: "Write a long story about a dragon.", options: GenerationOptions(maximumResponseTokens: 2048)) }
    try await Task.sleep(for: .milliseconds(300))
    task.cancel()
    _ = try? await task.value
    let after = try await cancelled.respond(to: "What is the capital of Japan? Answer with one word.", options: GenerationOptions(maximumResponseTokens: 512))
    #expect(cancelled.transcript.filter { if case .prompt = $0 { true } else { false } }.count == 1)
    #expect(stopReason(cancelled) == "stopToken")
    #expect(!after.content.isEmpty, "\(texts(cancelled))")
    #expect(await model.runtime.processed.count == after.usage.input.totalTokenCount + after.usage.output.totalTokenCount)

    let tiny = try await LocalLanguageModel(bundle: URL(filePath: path), contextLimit: 16)
    await #expect {
        try await LanguageModelSession(model: tiny).respond(to: "Tell me about the history of computing in detail.")
    } throws: { error in
        guard case LanguageModelError.contextSizeExceeded(let exceeded) = error else { return false }
        return exceeded.contextSize == 16 && exceeded.tokenCount > 16
    }
    #expect(await tiny.runtime.processed.isEmpty)
}

struct WeatherTool: Tool {
    let name = "get_weather"
    let description = "Get the current weather for a city."

    @Generable
    struct Arguments {
        @Guide(description: "The city name.")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        "Sunny, 22 C in \(arguments.city)."
    }
}

@Test(.enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
func sessionCallsToolsReusesAndRecovers(path: String) async throws {
    let model = try await LocalLanguageModel(bundle: URL(filePath: path))
    func metadata(_ session: LanguageModelSession, _ key: String) -> Int? {
        guard case .response(let response) = session.transcript.last else { return nil }
        return try? response.metadata[key]?.value(Int.self)
    }

    let session = LanguageModelSession(model: model, tools: [WeatherTool()])
    try await session.respond(to: "What is the weather in Tokyo right now?", options: GenerationOptions(maximumResponseTokens: 512))
    let calls = session.transcript.flatMap { entry -> [Transcript.ToolCall] in
        if case .toolCalls(let calls) = entry { Array(calls) } else { [] }
    }
    #expect(calls.map(\.toolName) == ["get_weather"])
    #expect(try calls.first?.arguments.value(String.self, forProperty: "city") == "Tokyo")
    #expect(session.transcript.contains { if case .toolOutput = $0 { true } else { false } })
    #expect(texts(session).contains("22"), "\(texts(session))")
    let shared = try #require(metadata(session, "sharedTokens"))
    let prefilled = try #require(metadata(session, "prefillTokens"))
    #expect(shared > 200 && prefilled < 64, "shared \(shared) prefilled \(prefilled)")
    #expect(metadata(session, "splicedGenerations") != nil)

    let tokenizer = model.runtime.tokenizer
    func encode(_ text: String) -> [Int32] { tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init) }
    let parts = ["\u{30EC}\u{30A4}", "\u{30C8}\u{30EC}\u{30FC}", "\u{30B7}\u{30F3}\u{30B0}"]
    let written = parts.flatMap(encode)
    let rerendered = encode(parts.joined())
    #expect(written != rerendered)
    let restored = ReusePlan.splice([0] + rerendered + [1], processed: [0] + written, generations: [(1, written)]) { tokenizer.decode(tokens: $0) }
    #expect(restored.prompt == [0] + written + [1] && restored.spliced == 1)

    let question = "What is the capital of Japan?"
    let allowed = try await LanguageModelSession(model: model, tools: [WeatherTool()]).respond(
        to: question, options: GenerationOptions(maximumResponseTokens: 16, toolCallingMode: .allowed))
    let disallowed = try await LanguageModelSession(model: model, tools: [WeatherTool()]).respond(
        to: question, options: GenerationOptions(maximumResponseTokens: 16, toolCallingMode: .disallowed))
    let plain = try await LanguageModelSession(model: model).respond(to: question, options: GenerationOptions(maximumResponseTokens: 16))
    #expect(disallowed.usage.input.totalTokenCount == plain.usage.input.totalTokenCount)
    #expect(allowed.usage.input.totalTokenCount > disallowed.usage.input.totalTokenCount + 100)
    let processed = await model.runtime.processed
    await #expect {
        try await LanguageModelSession(model: model, tools: [WeatherTool()]).respond(
            to: question, options: GenerationOptions(maximumResponseTokens: 16, toolCallingMode: .required))
    } throws: { error in
        guard case LanguageModelError.unsupportedCapability(let unsupported) = error else { return false }
        return unsupported.capability == .guidedGeneration
    }
    #expect(await model.runtime.processed == processed)

    let cut = LanguageModelSession(model: model, tools: [WeatherTool()])
    try await cut.respond(to: "What is the weather in Tokyo right now?", options: GenerationOptions(maximumResponseTokens: 8))
    #expect(!cut.transcript.contains { if case .toolCalls = $0 { true } else { false } })
    guard case .response(let last) = cut.transcript.last else { Issue.record("no response entry"); return }
    #expect(try last.metadata["stopReason"]?.value(String.self) == "maxTokens")

    let resumed = LanguageModelSession(model: model, instructions: "You are a concise assistant.")
    try await resumed.respond(to: "My name is Taro. Remember it.", options: GenerationOptions(maximumResponseTokens: 256))
    let task = Task { try await resumed.respond(to: "Write a long story about a dragon.", options: GenerationOptions(maximumResponseTokens: 2048)) }
    try await Task.sleep(for: .milliseconds(400))
    task.cancel()
    _ = try? await task.value
    let after = try await resumed.respond(to: "What is my name?", options: GenerationOptions(maximumResponseTokens: 256))
    #expect(after.usage.input.cachedTokenCount > 0, "\(after.usage)")
}
