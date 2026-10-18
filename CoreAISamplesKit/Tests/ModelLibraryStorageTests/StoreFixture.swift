import CryptoKit
import EmbeddingCore
import Foundation
import GLiNERRuntime
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

let distribution = "0123456789abcdef0123456789abcdef01234567"
let updatedDistribution = "fedcba9876543210fedcba9876543210fedcba98"
let glinerSource = ModelUpstream(repository: "fixture/gliner-source", revision: String(repeating: "a1", count: 20))
let embeddingSource = ModelUpstream(
    repository: "fixture/embedding-source", revision: "914f7f89142e33e77833254d9c9b90c3cef7303b")

private let glinerContract = #"""
    {
      "functions": {
        "encode_t256": {
          "inputs": ["input_ids", "attention_mask", "word_idx", "word_mask", "query_idx", "query_mask", "cls_idx",
                     "cls_mask"],
          "outputs": ["start_logits", "end_logits", "inside_prefix", "inside_mean", "null_logits", "cls_logits",
                      "pool_start", "pool_end", "scorer_start", "scorer_end", "content_prefix", "query_proj"]
        },
        "pair_t256": {
          "inputs": ["scorer_start", "scorer_end", "content_prefix", "query_proj", "start_logits", "end_logits",
                     "inside_prefix", "inside_mean", "word_mask", "query_mask", "cand_start", "cand_end", "cand_mask",
                     "cand_prior"],
          "outputs": ["pair_logits"]
        },
        "encode_t512": {
          "inputs": ["input_ids", "attention_mask", "word_idx", "word_mask", "query_idx", "query_mask", "cls_idx",
                     "cls_mask"],
          "outputs": ["start_logits", "end_logits", "inside_prefix", "inside_mean", "null_logits", "cls_logits",
                      "pool_start", "pool_end", "scorer_start", "scorer_end", "content_prefix", "query_proj"]
        },
        "pair_t512": {
          "inputs": ["scorer_start", "scorer_end", "content_prefix", "query_proj", "start_logits", "end_logits",
                     "inside_prefix", "inside_mean", "word_mask", "query_mask", "cand_start", "cand_end", "cand_mask",
                     "cand_prior"],
          "outputs": ["pair_logits"]
        }
      },
      "caps": {"tokens": [256, 512], "queries": 16, "class_labels": 16, "candidates": 192, "boundaries": "T + 1",
               "words": "T"},
      "host": {"pool_boundary_top_k": 32, "pool_size": 192, "min_pool_per_query": 8, "mask_logit": -10000.0,
               "quota_score_base": 5000.0, "pair_temperature": 1.0, "classification_temperature": 1.0,
               "abstention_threshold": 0.5, "overlap_policy": "flat", "default_threshold": 0.5,
               "token_pooling": "first", "append_period_unless_text_ends_with": [".", "!", "?"]}
    }
    """#

func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func describe(_ error: (any Error)?) -> String {
    error.map { String(reflecting: $0) } ?? "no error"
}

func identity(
    _ resource: ValidatedModelResource, modelID: String? = nil, contract: String? = nil, revision: String? = nil
) -> ModelArtifactIdentity {
    ValidatedModelResource(
        modelID: modelID ?? resource.modelID, artifactID: resource.artifactID, contract: contract ?? resource.contract,
        revision: revision ?? resource.revision, upstreamRevision: resource.upstreamRevision,
        directory: resource.directory, sizeBytes: resource.sizeBytes
    ).identity
}

struct Sample {
    let model: LibraryModel
    let artifact: ModelArtifact
    let files: [String: Data]

    init(
        _ id: String, model: String, contract: String, files: [String: Data], source: ModelUpstream?,
        revision: String = distribution, extra: [ModelArtifactFile] = []
    ) {
        self.files = files
        let defined =
            files.map { ModelArtifactFile(path: $0.key, bytes: Int64($0.value.count), sha256: sha256($0.value)) }
            .sorted { $0.path < $1.path } + extra
        artifact = ModelArtifact(
            id: id, title: id, contract: contract, platforms: [.iOS],
            sizeBytes: defined.reduce(0) { $0 + $1.bytes },
            download: ModelDownload(repository: "owner/fixture", revision: revision), upstream: source,
            files: defined)
        self.model = LibraryModel(
            id: model, name: model, category: "Fixture", artifacts: [artifact], defaultArtifactID: id,
            platforms: [.iOS], demos: [])
    }

    static func glinerFiles(
        source: ModelUpstream = glinerSource, edit: (inout [String: Any]) -> Void = { _ in }
    ) throws -> [String: Data] {
        var contract = try #require(JSONSerialization.jsonObject(with: Data(glinerContract.utf8)) as? [String: Any])
        contract["model"] = source.repository
        contract["revision"] = source.revision
        edit(&contract)
        return [
            "contract.json": try JSONSerialization.data(withJSONObject: contract, options: .sortedKeys),
            "x.aimodel/main.hash": Data("fixture-gliner-hash".utf8),
            "x.aimodel/main.mlirb": Data("fixture-gliner-weights".utf8),
            "x.aimodel/metadata.json": Data(#"{"fixture":"gliner"}"#.utf8),
            "tokenizer/tokenizer.json": Data(#"{"fixture":"tokenizer"}"#.utf8),
        ]
    }

    static func embeddingFiles(
        precision: String = "fp16-safe-s128-v70", revision: String = embeddingSource.revision
    ) throws -> [String: Data] {
        let tables =
            precision == "fp16-safe-s128-v70"
            ? ["token_embeddings.bf16"]
            : [
                "token_embeddings.int4", "token_embedding_scales.f32", "quantization-quality.json",
                "quantized-reference.json",
            ]
        var files: [String: Data] = [:]
        for name in [
            "text.aimodel/main.hash", "text.aimodel/main.mlirb", "text.aimodel/metadata.json",
            "vision.aimodel/main.hash", "vision.aimodel/main.mlirb", "vision.aimodel/metadata.json",
            "tokenizer.json", "tokenizer_config.json", "MODEL_LICENSE.txt",
        ] + tables {
            files[name] = Data("fixture-\(precision)-\(name)".utf8)
        }
        let manifest: [String: Any] = [
            "format": 1, "revision": revision, "precision": precision, "files": files.mapValues(sha256),
        ]
        files["embedding-manifest.json"] = try JSONSerialization.data(withJSONObject: manifest, options: .sortedKeys)
        return files
    }

    static func gliner(revision: String = distribution) throws -> Sample {
        Sample(
            "gliner-fp32", model: "gliner", contract: "gliner-gl4-fp32-t256-t512", files: try glinerFiles(),
            source: glinerSource, revision: revision)
    }

    static func embedding(int4: Bool = false) throws -> Sample {
        let precision = int4 ? "int4-int8-ptq-s128-v70" : "fp16-safe-s128-v70"
        return Sample(
            int4 ? "embedding-int4" : "embedding-fp16", model: "embedding", contract: "embeddinggemma2-" + precision,
            files: try embeddingFiles(precision: precision), source: embeddingSource)
    }

    func with(files: [String: Data]) -> Sample {
        Sample(
            artifact.id, model: model.id, contract: artifact.contract, files: files, source: artifact.upstream,
            revision: artifact.download?.revision ?? distribution)
    }
}

struct StoreFixture {
    let base = FileManager.default.temporaryDirectory.appending(path: "model-store-\(UUID().uuidString)")
    var root: URL { base.appending(path: "ModelLibrary") }
    var documents: URL { base.appending(path: "Documents") }

    func store(
        writeLedger: @escaping @Sendable (Data, URL) throws -> Void = { try $0.write(to: $1, options: .atomic) },
        availableCapacity: @escaping @Sendable (URL) throws -> Int64 = { _ in .max }
    ) -> ManagedModelStore {
        ManagedModelStore(
            root: root, clock: { Date(timeIntervalSince1970: 1_000) }, writeLedger: writeLedger,
            availableCapacity: availableCapacity)
    }

    func artifactDirectory(_ sample: Sample) -> URL {
        root.appending(path: "artifacts/\(sample.artifact.id)/\(sample.artifact.download?.revision ?? "")")
    }

    func stagingArea(_ sample: Sample) -> URL {
        root.appending(path: "staging/\(sample.artifact.id)/\(sample.artifact.download?.revision ?? "")")
    }

    func write(_ files: [String: Data], to directory: URL) throws {
        for (path, data) in files {
            let file = directory.appending(path: path)
            try FileManager.default.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: file)
        }
    }

    @discardableResult
    func install(_ sample: Sample, into store: ManagedModelStore) async throws -> ValidatedModelResource {
        let staging = try await store.stagingDirectory(for: sample.model, artifact: sample.artifact)
        try write(sample.files, to: staging)
        return try await store.promote(staging: staging, model: sample.model, artifact: sample.artifact)
    }

    func place(_ files: [String: Data], in directory: URL, linking link: String? = nil) throws {
        try write(files, to: directory)
        guard let link, let data = files[link] else { return }
        let target = base.appending(path: "link-targets/\(UUID().uuidString)")
        try write([link: data], to: target)
        try FileManager.default.removeItem(at: directory.appending(path: link))
        try FileManager.default.createSymbolicLink(
            at: directory.appending(path: link), withDestinationURL: target.appending(path: link))
    }

    func ledger() throws -> Ledger {
        try JSONDecoder().decode(Ledger.self, from: Data(contentsOf: root.appending(path: "ledger.json")))
    }

    func contents(_ directory: URL) -> [String: String] {
        var contents: [String: String] = [:]
        guard let entries = FileManager.default.enumerator(atPath: directory.path) else { return contents }
        while let path = entries.nextObject() as? String {
            let url = directory.appending(path: path)
            switch entries.fileAttributes?[.type] as? FileAttributeType {
            case .typeSymbolicLink?:
                contents[path] =
                    "link:" + ((try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) ?? "")
            case .typeRegular?: contents[path] = (try? Data(contentsOf: url)).map(sha256) ?? "unreadable"
            default: contents[path] = "directory"
            }
        }
        return contents
    }

    func remove() {
        if let entries = FileManager.default.enumerator(atPath: base.path) {
            while let path = entries.nextObject() as? String {
                guard entries.fileAttributes?[.type] as? FileAttributeType == .typeDirectory else { continue }
                try? FileManager.default.setAttributes(
                    [.posixPermissions: 0o755], ofItemAtPath: base.appending(path: path).path)
            }
        }
        try? FileManager.default.removeItem(at: base)
    }
}

extension ManagedInspection {
    var resource: ValidatedModelResource? { if case .ready(let resource) = self { resource } else { nil } }
    var reason: String? { if case .invalid(let error) = self { describe(error) } else { nil } }
    var isMissing: Bool { if case .missing = self { true } else { false } }
}

extension MigrationOutcome {
    var resource: ValidatedModelResource? { if case .migrated(let resource) = self { resource } else { nil } }
    var isSkipped: Bool { if case .skipped = self { true } else { false } }
    var undetermined: String? { if case .undetermined(let error) = self { describe(error) } else { nil } }
    var failure: String? { if case .failed(let error) = self { describe(error) } else { nil } }
}

enum Defect: String, CaseIterable, Sendable {
    case source, upstream, profile, contract, tokenizer, tokenizerHash, tableHash, size, missing, extra, link, escape
    case structure, manifest

    func make() throws -> (sample: Sample, files: [String: Data], link: String?, expected: any Error) {
        let gliner = try Sample.gliner()
        let embedding = try Sample.embedding()
        var files = gliner.files
        switch self {
        case .source:
            let other = ModelUpstream(repository: "fixture/other-source", revision: glinerSource.revision)
            let sample = gliner.with(files: try Sample.glinerFiles(source: other))
            return (sample, sample.files, nil, ManagedModelStoreError.contractMismatch("upstream"))
        case .upstream:
            let other = ModelUpstream(
                repository: embeddingSource.repository, revision: String(repeating: "b2", count: 20))
            let sample = Sample(
                embedding.artifact.id, model: embedding.model.id, contract: embedding.artifact.contract,
                files: embedding.files, source: other)
            return (sample, sample.files, nil, ManagedModelStoreError.contractMismatch("upstream"))
        case .profile:
            let sample = Sample(
                "embedding-int4", model: embedding.model.id, contract: "embeddinggemma2-int4-int8-ptq-s128-v70",
                files: embedding.files, source: embeddingSource)
            return (sample, sample.files, nil, ManagedModelStoreError.contractMismatch("precision"))
        case .contract:
            let sample = Sample(
                gliner.artifact.id, model: gliner.model.id, contract: "fixture-contract", files: files,
                source: glinerSource)
            return (sample, files, nil, ManagedModelStoreError.contractMismatch("contract"))
        case .tokenizer:
            files["tokenizer/tokenizer.json"] = nil
            return (gliner.with(files: files), files, nil, ManagedModelStoreError.contractMismatch("tokenizer"))
        case .tokenizerHash:
            files["tokenizer/tokenizer.json"] = Data(#"{"fixture":"tokenizeR"}"#.utf8)
            return (gliner, files, nil, ManagedModelStoreError.checksumMismatch("tokenizer/tokenizer.json"))
        case .tableHash:
            var changed = embedding.files
            changed["token_embeddings.bf16"] = Data(
                repeating: 0, count: embedding.files["token_embeddings.bf16"]?.count ?? 0)
            return (embedding, changed, nil, ManagedModelStoreError.checksumMismatch("token_embeddings.bf16"))
        case .size:
            files["x.aimodel/main.mlirb"] = Data("fixture".utf8)
            return (gliner, files, nil, ManagedModelStoreError.sizeMismatch("x.aimodel/main.mlirb"))
        case .missing:
            files["x.aimodel/metadata.json"] = nil
            return (gliner, files, nil, ManagedModelStoreError.missingFile("x.aimodel/metadata.json"))
        case .extra:
            files["x.aimodel/extra.bin"] = Data("extra".utf8)
            return (gliner, files, nil, ManagedModelStoreError.unexpectedFile("x.aimodel/extra.bin"))
        case .link:
            return (gliner, files, "tokenizer/tokenizer.json", ManagedModelStoreError.symbolicLink)
        case .escape:
            let escape = ModelArtifactFile(path: "../escape.json", bytes: 2, sha256: sha256(Data("{}".utf8)))
            let sample = Sample(
                gliner.artifact.id, model: gliner.model.id, contract: gliner.artifact.contract, files: files,
                source: glinerSource, extra: [escape])
            return (sample, files, nil, ManagedModelStoreError.pathEscape)
        case .structure:
            let sample = gliner.with(
                files: try Sample.glinerFiles { contract in
                    var caps = contract["caps"] as? [String: Any]
                    caps?["tokens"] = [256]
                    contract["caps"] = caps
                })
            return (sample, sample.files, nil, GLiNERAssetError.contractMismatch("caps"))
        case .manifest:
            let sample = embedding.with(files: try Sample.embeddingFiles(revision: String(repeating: "c3", count: 20)))
            return (sample, sample.files, nil, EmbeddingError.invalidBundle("Unsupported bundle contract"))
        }
    }
}

extension ManagedModelStore {
    func inspect(_ sample: Sample) throws -> ManagedInspection {
        try inspect(model: sample.model, artifact: sample.artifact)
    }

    func promote(_ staging: URL, _ sample: Sample) throws -> ValidatedModelResource {
        try promote(staging: staging, model: sample.model, artifact: sample.artifact)
    }

    func migrate(_ candidates: [URL], _ sample: Sample) -> MigrationReport {
        migrate(candidates: candidates, model: sample.model, artifact: sample.artifact)
    }

    func staging(_ sample: Sample) throws -> URL {
        try stagingDirectory(for: sample.model, artifact: sample.artifact)
    }
}
