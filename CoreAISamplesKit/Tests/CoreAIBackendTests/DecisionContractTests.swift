import CryptoKit
import Foundation
import Testing

@testable import CoreAIBackend

struct DecisionContractTests {
    @Test("SystemOne JSON preserves question order, scalar-distinct IDs and option descriptions")
    func orderedJSON() throws {
        let data = Data(
            #"{"state":"a\n\"b","questions":{"z":{"type":"choice","criteria":{"\u00e9":"first","e\u0301":null}},"a":{"type":"noul"},"rate":{"type":"score","criteria":["low",null,"high"]}}}"#
                .utf8)
        let request = try DecisionRequest(json: data)
        #expect(request.questions.map(\.id) == ["z", "a", "rate"])
        let roundtrip = try DecisionRequest(json: request.jsonData())
        #expect(request.jsonData() == roundtrip.jsonData())
        let rendered = try DecisionRenderer.render(request, limits: DecisionLimits()) { $0.utf8.map(Int.init) }
        #expect(rendered.optionIDs[0].map { Array($0.utf8) } == [Array("e\u{0301}".utf8), Array("\u{00e9}".utf8)])
        #expect(rendered.questionTypes == [1, 0, 2])
    }

    @Test(
        "Malformed JSON, duplicate keys and invalid question schemas are rejected",
        arguments: [
            #"{"state":"","questions":{"x":{"type":"noul"},"x":{"type":"noul"}}}"#,
            #"{"state":"","questions":{"x":{"type":"choice","criteria":{"a":"1","a":"2"}}}}"#,
            #"{"state":"","questions":{"x":{"type":"choice","criteria":{"a":12}}}}"#,
            #"{"state":"","questions":{"x":{"type":"noul","criteria":{"maybe":"yes"}}}}"#,
            #"{"state":"","questions":{"x":{"type":"unknown"}}}"#,
            #"{"state":true,"questions":{}}"#,
            #"{"state":"","questions":{},}"#,
            #"{"state":"\uZZZZ","questions":{}}"#,
            #"{"state":"","questions":{}} false"#,
        ])
    func invalidJSON(_ text: String) {
        #expect(throws: DecisionError.self) { try DecisionRequest(json: Data(text.utf8)) }
    }

    @Test("JSON arrays retain explicit question IDs and null instructions")
    func arrayJSON() throws {
        let request = try DecisionRequest(
            json: Data(#"{"state":"","questions":[{"id":"q","type":"noul","instructions":null}]}"#.utf8))
        #expect(request.questions == [DecisionQuestion(id: "q", criteria: .noul())])
    }

    @Test(
        "Decision bundles reject missing manifests, unsafe paths, bad contracts and corrupted hashes before specialization"
    )
    func bundleValidation() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(throws: DecisionError.self) { try DecisionBundle(url: folder) }
        let manifest = try makeBundle(at: folder)
        let bundle = try DecisionBundle(url: folder)
        #expect(bundle.limits.tokens == 9_216)
        #expect(bundle.name == "fixture")
        #expect(bundle.version.count == 64)
        for (old, new) in [
            ("clef-decision-bundle/1", "llm/1"),
            ("backbone/model.aimodel", "../model.aimodel"),
            ("\"inputTokens\":9216", "\"inputTokens\":99999"),
            ("\"hiddenSize\":4096", "\"hiddenSize\":2"),
        ] {
            try Data(manifest.replacingOccurrences(of: old, with: new).utf8)
                .write(to: folder.appendingPathComponent("decision.json"))
            #expect(throws: DecisionError.self) { try DecisionBundle(url: folder) }
        }
        try Data(manifest.utf8).write(to: folder.appendingPathComponent("decision.json"))
        let config = folder.appendingPathComponent("backbone/tokenizer/tokenizer_config.json")
        try Data("corrupt".utf8).write(to: config)
        #expect(throws: DecisionError.self) { try DecisionBundle(url: folder) }
        try Data("fixture".utf8).write(to: config)
        try Data("corrupt".utf8).write(to: folder.appendingPathComponent("backbone/model.aimodel/main.mlirb"))
        #expect(throws: DecisionError.self) { try DecisionBundle(url: folder) }
    }

    @Test("BF16 lexical rows are averaged in Float and rounded to Float16")
    func lexicalMean() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let url = folder.appendingPathComponent("lexical.safetensors")
        var header =
            #"{"__metadata__":{"format":"pt"},"lm_head.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#
        while !header.utf8.count.isMultiple(of: 8) { header += " " }
        var length = UInt64(header.utf8.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(Data(header.utf8))
        let values: [Float] = [1, 3, 5, 7]
        for value in values {
            var bits = UInt16(value.bitPattern >> 16).littleEndian
            data.append(withUnsafeBytes(of: &bits) { Data($0) })
        }
        try data.write(to: url)
        let lexical = try DecisionLexical(url: url, width: 2)
        let rendered = RenderedDecision(
            tokenIDs: [0, 1], questionSpans: [0..<1], optionSpans: [[0..<2]], optionIDs: [["a"]], questionTypes: [1])
        #expect(try lexical.vectors(rendered: rendered, questions: 1, options: 2) == [3, 5, 0, 0])
        let invalid = RenderedDecision(
            tokenIDs: [2], questionSpans: [0..<1], optionSpans: [[0..<1]], optionIDs: [["a"]], questionTypes: [1])
        #expect(throws: DecisionError.self) { try lexical.vectors(rendered: invalid, questions: 1, options: 2) }
    }

    @Test("Valid options alone determine choice, noul and score answers")
    func typedProbabilities() throws {
        let request = DecisionRequest(
            state: "",
            questions: [
                DecisionQuestion(id: "pick", criteria: .choice(["a": nil, "b": nil])),
                DecisionQuestion(id: "yes", criteria: .noul()),
                DecisionQuestion(id: "rate", criteria: .score(["low", "high"])),
            ])
        let rendered = try DecisionRenderer.render(request, limits: DecisionLimits()) { _ in [] }
        let logits: [Float] = [0, 0, 100, 0, 0, 100, 0, 0, 100]
        let answers = try DecisionResult.answers(request: request, rendered: rendered, logits: logits, stride: 3)
        #expect(answers.allSatisfy { $0.distribution.map(\.probability) == [0.5, 0.5] })
        #expect(answers[1] == .noul(id: "yes", distribution: answers[1].distribution, probabilityTrue: 0.5))
        #expect(answers[2] == .score(id: "rate", distribution: answers[2].distribution, expectedValue: 0.5))
        #expect(throws: DecisionError.self) {
            try DecisionResult.answers(
                request: request, rendered: rendered, logits: [.nan] + logits.dropFirst(), stride: 3)
        }
    }

    @Test("SystemOne responses preserve tied input order and expose type-specific answers and usage")
    func systemOneResponse() throws {
        let request = try DecisionRequest(
            json: Data(
                #"{"state":"","questions":{"pick":{"type":"choice","criteria":{"b":"B","a":"A"}},"yes":{"type":"noul"},"rate":{"type":"score","criteria":["low","high"]}}}"#
                    .utf8))
        let rendered = try DecisionRenderer.render(request, limits: DecisionLimits()) { _ in [] }
        let answers = try DecisionResult.answers(
            request: request, rendered: rendered, logits: Array(repeating: 0, count: 6), stride: 2)
        #expect(answers[0].top == "b")
        let result = DecisionResult(
            answers: answers, model: "fixture", modelVersion: "v1", inputTokens: 12, elapsedSeconds: 0.25)
        let root = try #require(
            try JSONSerialization.jsonObject(with: result.systemOneJSON(for: request)) as? [String: Any])
        let fields = try #require(root["answers"] as? [String: [String: Any]])
        #expect(fields["pick"]?["choice"] as? String == "b")
        #expect(fields["yes"]?["noul"] as? Double == 0.5)
        #expect(fields["rate"]?["score"] as? Double == 0.5)
        #expect((root["usage"] as? [String: Int]) == ["input_tokens": 12, "output_tokens": 0])
    }

    private func makeBundle(at root: URL) throws -> String {
        let paths = [
            "backbone/model.aimodel/main.mlirb", "head/model.aimodel/main.mlirb",
            "backbone/lexical.safetensors", "backbone/tokenizer/tokenizer.json", "backbone/metadata.json",
            "backbone/tokenizer/tokenizer_config.json",
        ]
        let bytes = Data("fixture".utf8)
        let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        for path in paths {
            let file = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: file)
        }
        let manifest = """
            {"format":"clef-decision-bundle/1","name":"fixture",
            "backbone":{"model":"backbone/model.aimodel","metadata":"backbone/metadata.json",
            "lexical":"backbone/lexical.safetensors","tokenizer":"backbone/tokenizer","function":"main",
            "inputs":["input_ids","position_ids"],"output":"hidden_states",
            "states":["keyCache","valueCache","convStates","recurrentStates"],"hiddenSize":4096,
            "sha256":{"main.mlirb":"\(hash)","lexical.safetensors":"\(hash)","tokenizer/tokenizer.json":"\(hash)"}},
            "head":{"model":"head/model.aimodel","function":"main",
            "inputs":["hidden","valid_len","question_spans","option_spans","question_types","lexical"],
            "output":"logits","precision":"fp16","sha256":{"main.mlirb":"\(hash)"}},
            "limits":{"questions":16,"options":8,"inputTokens":9216,"padMultiple":512},
            "rendering":{"version":1,"revision":"fixture"},"source":{"model":"fixture","revision":"fixture"}}
            """
        try Data(manifest.utf8).write(to: root.appendingPathComponent("decision.json"))
        var sums = paths.map { hash + "  " + $0 }
        let manifestHash = SHA256.hash(data: Data(manifest.utf8)).map { String(format: "%02x", $0) }.joined()
        sums.append(manifestHash + "  decision.json")
        try Data((sums.joined(separator: "\n") + "\n").utf8)
            .write(to: root.appendingPathComponent("SHA256SUMS"))
        return manifest
    }
}
