import CryptoKit
import Foundation

public struct DecisionBundle: Sendable {
    struct Manifest: Decodable, Sendable {
        struct Backbone: Decodable, Sendable {
            let model: String
            let metadata: String
            let lexical: String
            let tokenizer: String
            let function: String
            let inputs: [String]
            let output: String
            let states: [String]
            let hiddenSize: Int
            let sha256: [String: String]
        }

        struct Head: Decodable, Sendable {
            let model: String
            let function: String
            let inputs: [String]
            let output: String
            let precision: String
            let sha256: [String: String]
        }

        struct Limits: Decodable, Sendable {
            let questions: Int
            let options: Int
            let inputTokens: Int
            let padMultiple: Int
        }

        struct Rendering: Decodable, Sendable {
            let version: Int
            let revision: String
        }

        struct Source: Decodable, Sendable {
            let model: String
            let revision: String
        }

        let format: String
        let name: String
        let backbone: Backbone
        let head: Head
        let limits: Limits
        let rendering: Rendering
        let source: Source
    }

    public let url: URL
    public let name: String
    public let limits: DecisionLimits
    public let version: String
    let manifest: Manifest
    let backboneURL: URL
    let headURL: URL
    let lexicalURL: URL
    let tokenizerURL: URL

    public init(url: URL) throws {
        do {
            let root = url.standardizedFileURL.resolvingSymlinksInPath()
            let data = try Data(contentsOf: root.appendingPathComponent("decision.json"))
            let manifest = try JSONDecoder().decode(Manifest.self, from: data)
            guard manifest.format == "clef-decision-bundle/1",
                manifest.rendering.version == 1,
                manifest.backbone.hiddenSize == 4_096,
                manifest.backbone.inputs == ["input_ids", "position_ids"],
                manifest.backbone.output == "hidden_states",
                Set(manifest.backbone.states) == Set(["keyCache", "valueCache", "convStates", "recurrentStates"]),
                manifest.head.inputs == [
                    "hidden", "valid_len", "question_spans", "option_spans", "question_types", "lexical",
                ],
                manifest.head.output == "logits", manifest.head.precision == "fp16",
                manifest.limits.questions == 16, manifest.limits.options == 8,
                manifest.limits.inputTokens == 9_216, manifest.limits.padMultiple == 512
            else { throw DecisionError.invalidBundle("Unsupported decision bundle contract.") }
            let backbone = try Self.resolve(manifest.backbone.model, in: root)
            let head = try Self.resolve(manifest.head.model, in: root)
            let lexical = try Self.resolve(manifest.backbone.lexical, in: root)
            let tokenizer = try Self.resolve(manifest.backbone.tokenizer, in: root)
            _ = try Self.resolve(manifest.backbone.metadata, in: root)
            guard let backboneHash = manifest.backbone.sha256["main.mlirb"],
                let headHash = manifest.head.sha256["main.mlirb"],
                let lexicalHash = manifest.backbone.sha256["lexical.safetensors"],
                let tokenizerHash = manifest.backbone.sha256["tokenizer/tokenizer.json"]
            else { throw DecisionError.invalidBundle("The manifest is missing a required SHA-256.") }
            let declared = [
                (try Self.resolve(manifest.backbone.model + "/main.mlirb", in: root), backboneHash),
                (try Self.resolve(manifest.head.model + "/main.mlirb", in: root), headHash),
                (lexical, lexicalHash),
                (try Self.resolve(manifest.backbone.tokenizer + "/tokenizer.json", in: root), tokenizerHash),
            ]
            let sumsURL = try Self.resolve("SHA256SUMS", in: root)
            let sums = try String(contentsOf: sumsURL, encoding: .utf8)
            var expectedFiles: [URL: String] = [:]
            for line in sums.split(whereSeparator: \.isNewline) {
                let parts = line.split(maxSplits: 1, whereSeparator: \.isWhitespace)
                guard parts.count == 2 else { throw DecisionError.invalidBundle("Invalid SHA256SUMS entry.") }
                let file = try Self.resolve(String(parts[1]).trimmingCharacters(in: .whitespaces), in: root)
                guard expectedFiles.updateValue(String(parts[0]).lowercased(), forKey: file) == nil else {
                    throw DecisionError.invalidBundle("Repeated SHA256SUMS path.")
                }
            }
            for (file, expected) in declared {
                guard expectedFiles[file] == expected.lowercased() else {
                    throw DecisionError.invalidBundle("The manifest and SHA256SUMS disagree.")
                }
            }
            for path in [
                "decision.json", manifest.backbone.metadata,
                manifest.backbone.tokenizer + "/tokenizer_config.json",
            ] {
                guard expectedFiles[try Self.resolve(path, in: root)] != nil else {
                    throw DecisionError.invalidBundle("SHA256SUMS omits a required file: \(path).")
                }
            }
            for (file, expected) in expectedFiles {
                guard expected.count == 64, expected.allSatisfy(\.isHexDigit),
                    try Self.digest(file) == expected
                else { throw DecisionError.invalidBundle("SHA-256 mismatch: \(file.lastPathComponent).") }
            }
            self.url = root
            self.name = manifest.name
            self.manifest = manifest
            self.version = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            self.limits = DecisionLimits(
                questions: manifest.limits.questions, optionsPerQuestion: manifest.limits.options,
                tokens: manifest.limits.inputTokens)
            backboneURL = backbone
            headURL = head
            lexicalURL = lexical
            tokenizerURL = tokenizer
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as DecisionError {
            throw error
        } catch {
            throw DecisionError.invalidBundle(error.localizedDescription)
        }
    }

    private static func resolve(_ path: String, in root: URL) throws -> URL {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
            throw DecisionError.invalidBundle("Invalid relative path: \(path).")
        }
        let url = root.appendingPathComponent(path).standardizedFileURL.resolvingSymlinksInPath()
        guard url.path.hasPrefix(root.path + "/"), FileManager.default.fileExists(atPath: url.path) else {
            throw DecisionError.invalidBundle("Missing or out-of-bundle path: \(path).")
        }
        return url
    }

    private static func digest(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        while let data = try file.read(upToCount: 4 * 1_024 * 1_024), !data.isEmpty {
            try Task.checkCancellation()
            hash.update(data: data)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
