import CryptoKit
import Foundation

public struct TypedDecisionLimits: Sendable, Hashable {
    public let options: Int
    public let inputTokens: Int
}

public struct TypedDecisionBundle: Sendable {
    struct Document: Decodable {
        struct Backbone: Decodable {
            struct Hidden: Decodable {
                let dtype: String
                let size: Int
            }

            enum CodingKeys: String, CodingKey {
                case asset, function, inputs, output, hidden, states
                case contextLength = "context_length"
            }

            let asset: String
            let function: String
            let inputs: [String]
            let output: String
            let hidden: Hidden
            let states: [String]
            let contextLength: Int
        }

        struct Head: Decodable {
            enum CodingKeys: String, CodingKey {
                case file, weight, bias, dtype, sha256
                case weightShape = "weight_shape"
                case biasShape = "bias_shape"
            }

            let file: String
            let weight: String
            let bias: String
            let weightShape: [Int]
            let biasShape: [Int]
            let dtype: String
            let sha256: String
        }

        struct Temperatures: Decodable {
            enum CodingKeys: String, CodingKey { case perKind = "per_kind" }
            let perKind: [String: Double]
        }

        struct Limits: Decodable {
            enum CodingKeys: String, CodingKey {
                case options
                case inputTokens = "input_tokens"
            }

            let options: Int
            let inputTokens: Int
        }

        struct File: Decodable {
            let path: String
            let bytes: Int
            let sha256: String
        }

        enum CodingKeys: String, CodingKey {
            case format, backbone, head, slots, verbalizers, temperatures, limits, ties, files
            case verbalizerIDs = "verbalizer_ids"
        }

        let format: String
        let backbone: Backbone
        let head: Head
        let slots: [String: [Int]]
        let verbalizers: [String]
        let verbalizerIDs: [Int]
        let temperatures: Temperatures
        let limits: Limits
        let ties: String
        let files: [File]
    }

    static let states = ["keyCache", "valueCache", "convStates", "recurrentStates"]

    public let url: URL
    public let name: String
    public let version: String
    public let hiddenSize: Int
    public let limits: TypedDecisionLimits
    public let temperatures: [TypedDecisionKind: Float]
    public let verbalizerIDs: [Int]
    public let headURL: URL
    public let tokenizerURL: URL
    public let assetURL: URL
    public let backboneFunction: String
    public let backboneInputs: [String]
    public let backboneOutput: String
    public let stateNames: [String]
    public let contextLength: Int

    public init(url: URL) throws {
        do {
            let root = url.standardizedFileURL.resolvingSymlinksInPath()
            let data = try Data(contentsOf: root.appending(path: "decision.json"))
            try Self.checkKeys(data)
            let document = try JSONDecoder().decode(Document.self, from: data)
            let temperatures = try Self.validate(document)
            let version = Self.hex(SHA256.hash(data: data))
            try Self.checkFiles(document, root: root, version: version)
            self.url = root
            self.name = root.lastPathComponent
            self.version = version
            self.hiddenSize = document.backbone.hidden.size
            self.limits = TypedDecisionLimits(
                options: document.limits.options, inputTokens: document.limits.inputTokens)
            self.temperatures = temperatures
            self.verbalizerIDs = document.verbalizerIDs
            self.headURL = root.appending(path: document.head.file)
            self.tokenizerURL = root.appending(path: "tokenizer")
            self.assetURL = root.appending(path: document.backbone.asset)
            self.backboneFunction = document.backbone.function
            self.backboneInputs = document.backbone.inputs
            self.backboneOutput = document.backbone.output
            self.stateNames = document.backbone.states
            self.contextLength = document.backbone.contextLength
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as TypedDecisionError {
            throw error
        } catch {
            throw TypedDecisionError.invalidBundle(error.localizedDescription)
        }
    }

    private static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw TypedDecisionError.invalidBundle(message) }
    }

    private static func checkKeys(_ data: Data) throws {
        let document = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let keys = { (value: Any?) in (value as? [String: Any]).map { Set($0.keys) } }
        let temperatures = document["temperatures"] as? [String: Any]
        try require(
            document["source"] is [String: Any]
                && keys((document["backbone"] as? [String: Any])?["hidden"]) == ["dtype", "size"]
                && keys(document["limits"]) == ["options", "input_tokens"]
                && (document["files"] as? [Any])?.allSatisfy { keys($0) == ["path", "bytes", "sha256"] } == true
                && ["calibration_id", "calibration_sha256", "source"].allSatisfy { temperatures?[$0] is String },
            "decision.json has missing or unexpected keys.")
        let template: [String: Any] = [
            "version": "bare-v1", "text": TypedDecisionRenderer.template, "option_separator": "\n",
            "choice_line": "{label}) {option}", "labels": TypedDecisionRenderer.labels,
            "fixed_options": Dictionary(
                uniqueKeysWithValues: TypedDecisionRenderer.fixedOptions.map { ($0.key.rawValue, $0.value) }),
            "add_special_tokens": false, "last_token": "]:",
        ]
        try require(
            (document["template"] as? NSDictionary)?.isEqual(to: template) == true, "template differs from bare-v1.")
    }

    private static func validate(_ document: Document) throws -> [TypedDecisionKind: Float] {
        let backbone = document.backbone
        let head = document.head
        try require(document.format == "typed-decision-v1", "format must be typed-decision-v1.")
        try require(
            backbone.function == "main" && backbone.inputs == ["input_ids", "position_ids"]
                && backbone.output == "hidden_states" && backbone.states == states && backbone.contextLength > 0
                && backbone.hidden.dtype == "float16" && backbone.hidden.size > 0 && isRelative(backbone.asset),
            "backbone must describe main with the hidden-state contract.")
        try require(
            head.weight == "proj.weight" && head.bias == "proj.bias" && head.dtype == "float32"
                && head.weightShape == [TypedDecisionHead.rows, backbone.hidden.size]
                && head.biasShape == [TypedDecisionHead.rows] && isDigest(head.sha256) && isRelative(head.file),
            "head must hold proj.weight [24, hidden] and proj.bias [24] in float32 with a SHA-256.")
        try require(
            document.slots == ["noul": [0, 2], "score": [2, 8], "choice": [8, 24]],
            "slots must be noul [0, 2), score [2, 8) and choice [8, 24).")
        try require(
            document.verbalizers.count == TypedDecisionHead.rows
                && document.verbalizerIDs.count == TypedDecisionHead.rows,
            "verbalizers must name every head row.")
        var temperatures: [TypedDecisionKind: Float] = [:]
        for (key, value) in document.temperatures.perKind {
            if let kind = TypedDecisionKind(rawValue: key), Float(value).isFinite, Float(value) > 0 {
                temperatures[kind] = Float(value)
            }
        }
        try require(
            temperatures.count == TypedDecisionKind.allCases.count
                && document.temperatures.perKind.count == temperatures.count,
            "temperatures.per_kind must give every kind a positive finite value.")
        try require(
            (1...TypedDecisionRenderer.labels.count).contains(document.limits.options)
                && document.limits.inputTokens > 0,
            "limits must give 1 to 16 options and a positive input_tokens.")
        try require(document.ties == "lowest-index", "ties must be lowest-index.")
        for file in document.files {
            try require(
                isRelative(file.path) && file.path != "decision.json" && file.path != "SHA256SUMS" && file.bytes >= 0
                    && isDigest(file.sha256),
                "files has an invalid entry: \(file.path).")
        }
        try require(
            zip(document.files, document.files.dropFirst()).allSatisfy {
                $0.path.utf8.lexicographicallyPrecedes($1.path.utf8)
            },
            "files must be unique and sorted by path.")
        return temperatures
    }

    private static func checkFiles(_ document: Document, root: URL, version: String) throws {
        let files = document.files
        let paths = files.map { Array($0.path.utf8) }
        let listed = { (path: String) in paths.contains(Array(path.utf8)) }
        try require(
            files.contains {
                Array($0.path.utf8) == Array(document.head.file.utf8) && $0.sha256 == document.head.sha256
            },
            "head.file must be listed with the head digest.")
        try require(
            files.contains { $0.path.utf8.starts(with: (document.backbone.asset + "/").utf8) },
            "backbone.asset must be listed.")
        for path in ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
            try require(listed(path), "files must list \(path).")
        }
        let manager = FileManager.default
        var found: [[UInt8]] = []
        for path in try manager.subpathsOfDirectory(atPath: root.path) {
            let type =
                try manager.attributesOfItem(atPath: root.appending(path: path).path)[.type] as? FileAttributeType
            try require(type != .typeSymbolicLink, "Bundle files must not be links: \(path).")
            if type == .typeRegular, path != "decision.json", path != "SHA256SUMS" {
                found.append(Array(path.utf8))
            }
        }
        try require(
            found.sorted { $0.lexicographicallyPrecedes($1) } == paths, "files differ from the files in the bundle.")
        var sums: [[UInt8]: String] = [:]
        for line in try Data(contentsOf: root.appending(path: "SHA256SUMS")).split(separator: UInt8(ascii: "\n")) {
            let bytes = Array(line)
            let digest = String(decoding: bytes.prefix(64), as: UTF8.self)
            try require(
                bytes.count > 66 && bytes[64] == 0x20 && bytes[65] == 0x20 && isDigest(digest),
                "SHA256SUMS has an invalid line.")
            try require(sums.updateValue(digest, forKey: Array(bytes[66...])) == nil, "SHA256SUMS repeats a path.")
        }
        let expected = Dictionary(uniqueKeysWithValues: zip(paths, files.map(\.sha256)))
            .merging([Array("decision.json".utf8): version]) { first, _ in first }
        try require(sums == expected, "SHA256SUMS differs from decision.json.")
        for file in files {
            let url = root.appending(path: file.path)
            let size = try manager.attributesOfItem(atPath: url.path)[.size] as? NSNumber
            try require(size?.intValue == file.bytes, "Size mismatch: \(file.path).")
        }
        for file in files {
            try require(try digest(root.appending(path: file.path)) == file.sha256, "SHA-256 mismatch: \(file.path).")
        }
    }

    private static func isRelative(_ path: String) -> Bool {
        let parts = path.utf8.split(separator: UInt8(ascii: "/"), omittingEmptySubsequences: false)
        return !path.utf8.contains(0)
            && parts.allSatisfy { !$0.isEmpty && !$0.elementsEqual(".".utf8) && !$0.elementsEqual("..".utf8) }
    }

    private static func isDigest(_ text: String) -> Bool {
        text.utf8.count == 64 && text.utf8.allSatisfy { (0x30...0x39).contains($0) || (0x61...0x66).contains($0) }
    }

    private static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func digest(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        while try autoreleasepool(invoking: {
            try Task.checkCancellation()
            guard let data = try file.read(upToCount: 8 * 1_024 * 1_024), !data.isEmpty else { return false }
            hash.update(data: data)
            return true
        }) {}
        return hex(hash.finalize())
    }
}
