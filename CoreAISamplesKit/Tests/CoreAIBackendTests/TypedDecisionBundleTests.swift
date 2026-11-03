import CryptoKit
import Foundation
import Testing

@testable import CoreAIBackend

private func hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

enum TypedDecisionFakeBundle {
    static let files: [String: Data] = [
        "fixture.aimodel/main.mlirb": Data("main".utf8),
        "head.safetensors": TypedDecisionHeadFile.data(
            weight: Array(repeating: [0.5, -0.5, 0.25, 0], count: 24), bias: Array(repeating: 0, count: 24)),
        "tokenizer/tokenizer.json": Data("{}".utf8),
        "tokenizer/tokenizer_config.json": Data("{}".utf8),
    ]

    static func create() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString).appending(
            path: "jev27b_fixture")
        for (path, data) in files {
            let url = root.appending(path: path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url)
        }
        return root
    }

    static func document() -> [String: Any] {
        let entries = files.keys.sorted { $0.utf8.lexicographicallyPrecedes($1.utf8) }.map { path in
            ["path": path, "bytes": files[path]?.count ?? 0, "sha256": hex(files[path] ?? Data())] as [String: Any]
        }
        return [
            "format": "typed-decision-v1",
            "backbone": [
                "asset": "fixture.aimodel", "function": "main", "inputs": ["input_ids", "position_ids"],
                "output": "hidden_states", "hidden": ["dtype": "float16", "size": 4],
                "states": ["keyCache", "valueCache", "convStates", "recurrentStates"], "context_length": 4_096,
            ] as [String: Any],
            "head": [
                "file": "head.safetensors", "weight": "proj.weight", "bias": "proj.bias", "weight_shape": [24, 4],
                "bias_shape": [24], "dtype": "float32", "sha256": hex(files["head.safetensors"] ?? Data()),
            ] as [String: Any],
            "slots": ["noul": [0, 2], "score": [2, 8], "choice": [8, 24]],
            "verbalizers": (0..<24).map { "v\($0)" },
            "verbalizer_ids": Array(0..<24),
            "template": [
                "version": "bare-v1",
                "text": "[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n{options}\n[decision]:",
                "option_separator": "\n", "choice_line": "{label}) {option}", "labels": "ABCDEFGHIJKLMNOP",
                "fixed_options": ["noul": ["false", "true"], "score": ["0", "1", "2", "3", "4", "5"]],
                "add_special_tokens": false, "last_token": "]:",
            ] as [String: Any],
            "temperatures": [
                "per_kind": ["noul": 1.25, "score": 1.5, "choice": 0.75], "calibration_id": "fixture",
                "calibration_sha256": String(repeating: "0", count: 64), "source": "card",
            ] as [String: Any],
            "limits": ["options": 16, "input_tokens": 8_192],
            "ties": "lowest-index",
            "source": ["model": "fixture"],
            "files": entries,
        ]
    }

    static func setting(_ value: Any?, at path: [String], in object: [String: Any]) -> [String: Any] {
        var object = object
        guard let key = path.first else { return object }
        object[key] =
            path.count == 1
            ? value : setting(value, at: Array(path.dropFirst()), in: object[key] as? [String: Any] ?? [:])
        return object
    }

    static func entries(_ document: [String: Any]) -> [[String: Any]] {
        document["files"] as? [[String: Any]] ?? []
    }

    static func write(_ document: [String: Any], to root: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: document, options: [.sortedKeys, .withoutEscapingSlashes])
        try data.write(to: root.appending(path: "decision.json"))
        let lines = entries(document).compactMap { entry -> String? in
            guard let path = entry["path"] as? String, let digest = entry["sha256"] as? String else { return nil }
            return "\(digest)  \(path)\n"
        }
        let sums = (lines + ["\(hex(data))  decision.json\n"]).joined()
        try Data(sums.utf8).write(to: root.appending(path: "SHA256SUMS"))
    }
}

struct TypedDecisionRefit: Decodable {
    let temperatures: [String: Double]
}

struct TypedDecisionBundleDefect: Sendable, CustomTestStringConvertible {
    let testDescription: String
    let fragment: String
    var change: @Sendable ([String: Any]) -> [String: Any] = { $0 }
    var disk: @Sendable (URL) throws -> Void = { _ in }

    init(_ name: String, _ fragment: String, at path: [String], _ value: (any Sendable)?) {
        testDescription = name
        self.fragment = fragment
        change = { TypedDecisionFakeBundle.setting(value, at: path, in: $0) }
    }

    init(_ name: String, _ fragment: String, files: @escaping @Sendable ([[String: Any]]) -> [[String: Any]]) {
        testDescription = name
        self.fragment = fragment
        change = { $0.merging(["files": files(TypedDecisionFakeBundle.entries($0))]) { _, new in new } }
    }

    init(_ name: String, _ fragment: String, disk: @escaping @Sendable (URL) throws -> Void) {
        testDescription = name
        self.fragment = fragment
        self.disk = disk
    }

    static func updating(_ path: String, _ key: String, to value: any Sendable)
        -> @Sendable ([[String: Any]]) -> [[String: Any]]
    {
        { entries in entries.map { $0["path"] as? String == path ? $0.merging([key: value]) { _, new in new } : $0 } }
    }

    static let all: [TypedDecisionBundleDefect] = [
        .init("format", "format must be typed-decision-v1", at: ["format"], "typed-decision-v2"),
        .init("weight shape", "head must hold", at: ["head", "weight_shape"], [24, 5]),
        .init("bias shape", "head must hold", at: ["head", "bias_shape"], [23]),
        .init("hidden size", "head must hold", at: ["backbone", "hidden", "size"], 5),
        .init("hidden dtype", "backbone must describe", at: ["backbone", "hidden", "dtype"], "float32"),
        .init(
            "state order", "backbone must describe", at: ["backbone", "states"],
            ["valueCache", "keyCache", "convStates", "recurrentStates"]),
        .init("context length", "backbone must describe", at: ["backbone", "context_length"], 0),
        .init("zero temperature", "temperatures.per_kind", at: ["temperatures", "per_kind", "noul"], 0.0),
        .init("negative temperature", "temperatures.per_kind", at: ["temperatures", "per_kind", "score"], -1.5),
        .init("temperature below Float", "temperatures.per_kind", at: ["temperatures", "per_kind", "choice"], 1e-60),
        .init("missing temperature", "temperatures.per_kind", at: ["temperatures", "per_kind", "choice"], nil),
        .init("slots", "slots must be", at: ["slots", "choice"], [8, 23]),
        .init("too many options", "limits must", at: ["limits", "options"], 17),
        .init("no input tokens", "limits must", at: ["limits", "input_tokens"], 0),
        .init("ties", "ties must be", at: ["ties"], "highest-index"),
        .init("verbalizer IDs", "verbalizers must", at: ["verbalizer_ids"], Array(0..<23)),
        .init(
            "template text", "template differs", at: ["template", "text"],
            "[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n{options}\n[decision]"),
        .init("template labels", "template differs", at: ["template", "labels"], "ABCDEFGHIJ\u{212A}LMNOP"),
        .init("fixed options", "template differs", at: ["template", "fixed_options", "noul"], ["true", "false"]),
        .init("template key", "template differs", at: ["template", "extra"], "x"),
        .init("missing source", "missing or unexpected keys", at: ["source"], nil),
        .init("missing calibration", "missing or unexpected keys", at: ["temperatures", "calibration_sha256"], nil),
        .init("hidden key", "missing or unexpected keys", at: ["backbone", "hidden", "extra"], 1),
        .init("limits key", "missing or unexpected keys", at: ["limits", "extra"], 1),
        .init("file entry key", "missing or unexpected keys", files: updating("head.safetensors", "extra", to: 1)),
        .init("head digest", "head.file must be listed", at: ["head", "sha256"], String(repeating: "0", count: 64)),
        .init("asset", "backbone.asset must be listed", at: ["backbone", "asset"], "other.aimodel"),
        .init(
            "absolute path", "files has an invalid entry",
            files: updating("head.safetensors", "path", to: "/head.safetensors")),
        .init(
            "parent path", "files has an invalid entry",
            files: updating("tokenizer/tokenizer.json", "path", to: "tokenizer/../tokenizer.json")),
        .init("unsorted files", "files must be unique and sorted", files: { $0.reversed() }),
        .init("repeated file", "files must be unique and sorted", files: { Array($0.prefix(1)) + $0 }),
        .init(
            "missing tokenizer config", "files must list tokenizer/tokenizer_config.json",
            files: { $0.filter { $0["path"] as? String != "tokenizer/tokenizer_config.json" } }),
        .init(
            "listed file missing from the bundle", "files differ",
            files: { $0 + [["path": "zz.bin", "bytes": 0, "sha256": String(repeating: "0", count: 64)]] }),
        .init(
            "bundle file missing from files", "files differ",
            disk: { try Data("extra".utf8).write(to: $0.appending(path: "extra.txt")) }),
        .init("size", "Size mismatch", files: updating("tokenizer/tokenizer.json", "bytes", to: 3)),
        .init(
            "file content", "SHA-256 mismatch: fixture.aimodel/main.mlirb",
            disk: { try Data("MAIN".utf8).write(to: $0.appending(path: "fixture.aimodel/main.mlirb")) }),
        .init(
            "SHA256SUMS entry", "SHA256SUMS differs",
            disk: { root in
                let url = root.appending(path: "SHA256SUMS")
                let text = try String(contentsOf: url, encoding: .utf8)
                try Data(((text.first == "0" ? "1" : "0") + text.dropFirst()).utf8).write(to: url)
            }),
        .init(
            "SHA256SUMS without decision.json", "SHA256SUMS differs",
            disk: { root in
                let url = root.appending(path: "SHA256SUMS")
                let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n")
                try Data(lines.filter { !$0.hasSuffix("  decision.json") }.map { $0 + "\n" }.joined().utf8).write(
                    to: url)
            }),
        .init(
            "link", "must not be links",
            disk: { root in
                try FileManager.default.createSymbolicLink(
                    at: root.appending(path: "tokenizer/link.json"),
                    withDestinationURL: root.appending(path: "SHA256SUMS"))
            }),
        .init(
            "malformed decision.json", "", disk: { try Data("{".utf8).write(to: $0.appending(path: "decision.json")) }),
        .init(
            "missing decision.json", "",
            disk: { try FileManager.default.removeItem(at: $0.appending(path: "decision.json")) }),
    ]
}

struct TypedDecisionBundleTests {
    @Test("A minimal bundle opens and exposes its contract")
    func aMinimalBundleOpens() throws {
        let root = try TypedDecisionFakeBundle.create()
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        try TypedDecisionFakeBundle.write(TypedDecisionFakeBundle.document(), to: root)
        let bundle = try TypedDecisionBundle(url: root)
        #expect(bundle.url.path == root.standardizedFileURL.resolvingSymlinksInPath().path)
        #expect(bundle.name == "jev27b_fixture")
        #expect(bundle.version == hex(try Data(contentsOf: root.appending(path: "decision.json"))))
        #expect(bundle.hiddenSize == 4)
        #expect(bundle.limits == TypedDecisionLimits(options: 16, inputTokens: 8_192))
        #expect(bundle.temperatures == [.noul: 1.25, .score: 1.5, .choice: 0.75])
        #expect(bundle.verbalizerIDs == Array(0..<24))
        #expect(bundle.backboneFunction == "main")
        #expect(bundle.backboneInputs == ["input_ids", "position_ids"])
        #expect(bundle.backboneOutput == "hidden_states")
        #expect(bundle.stateNames == ["keyCache", "valueCache", "convStates", "recurrentStates"])
        #expect(bundle.contextLength == 4_096)
        #expect(bundle.headURL == bundle.url.appending(path: "head.safetensors"))
        #expect(bundle.tokenizerURL == bundle.url.appending(path: "tokenizer"))
        #expect(bundle.assetURL == bundle.url.appending(path: "fixture.aimodel"))
        let head = try TypedDecisionHead(url: bundle.headURL, width: bundle.hiddenSize)
        #expect(try head.logits(hidden: [1, 1, 1, 1]) == Array(repeating: 0.25, count: 24))
    }

    @Test(
        "A decision.json or bundle that breaks the contract is refused with invalidBundle",
        arguments: TypedDecisionBundleDefect.all)
    func brokenBundlesAreRefused(_ defect: TypedDecisionBundleDefect) throws {
        let root = try TypedDecisionFakeBundle.create()
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        try TypedDecisionFakeBundle.write(defect.change(TypedDecisionFakeBundle.document()), to: root)
        try defect.disk(root)
        let error = #expect(throws: TypedDecisionError.self) { try TypedDecisionBundle(url: root) }
        guard case .invalidBundle(let message)? = error else {
            Issue.record("Expected invalidBundle, got \(String(describing: error))")
            return
        }
        #expect(defect.fragment.isEmpty || message.contains(defect.fragment), "\(message)")
    }

    @Test(
        "The exported JEV-27B bundle opens after checking the size and SHA-256 of every listed file",
        .enabled(if: !jevBundleFolder.isEmpty))
    func theExportedBundleOpens() throws {
        let clock = ContinuousClock()
        var opened: TypedDecisionBundle?
        let elapsed = try clock.measure { opened = try TypedDecisionBundle(url: URL(filePath: jevBundleFolder)) }
        let bundle = try #require(opened)
        print("bundle \(bundle.name) opened in \(elapsed) version \(bundle.version)")
        #expect(bundle.version.count == 64 && bundle.version.allSatisfy(\.isHexDigit))
        #expect(bundle.hiddenSize == 5_120)
        #expect(bundle.limits == TypedDecisionLimits(options: 16, inputTokens: 8_192))
        #expect(bundle.verbalizerIDs.count == 24)
        #expect(bundle.backboneFunction == "main" && bundle.backboneOutput == "hidden_states")
        #expect(bundle.backboneInputs == ["input_ids", "position_ids"])
        #expect(bundle.stateNames == ["keyCache", "valueCache", "convStates", "recurrentStates"])
        #expect(bundle.contextLength == 262_144)
        let refit = bundle.url.appending(path: "calibration-refit.json")
        let expected =
            FileManager.default.fileExists(atPath: refit.path(percentEncoded: false))
            ? try JSONDecoder().decode(TypedDecisionRefit.self, from: Data(contentsOf: refit)).temperatures
            : try TypedDecisionRealHead.load().temperatures
        let temperatures = expected.compactMap { key, value in
            TypedDecisionKind(rawValue: key).map { ($0, Float(value)) }
        }
        #expect(temperatures.count == 3)
        #expect(bundle.temperatures == Dictionary(uniqueKeysWithValues: temperatures))
        _ = try TypedDecisionHead(url: bundle.headURL, width: bundle.hiddenSize)
    }
}
