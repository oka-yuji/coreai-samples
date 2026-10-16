import CoreAI
import EmbeddingCore
import Foundation

public actor EmbeddingModel {
    private struct Loaded {
        let text: InferenceFunction
        let vision: InferenceFunction
        let tokenizer: EmbeddingTokenizer
        let table: TokenEmbeddingTable
    }

    private var loaded: Loaded?
    private var directory: URL?
    private var busy = false
    private let beforeLoad: @Sendable () async throws -> Void

    public init() { beforeLoad = {} }

    init(beforeLoad: @escaping @Sendable () async throws -> Void) {
        self.beforeLoad = beforeLoad
    }

    deinit { directory?.stopAccessingSecurityScopedResource() }

    public func load(directory: URL) async throws {
        try begin()
        defer { busy = false }
        try await beforeLoad()
        try Task.checkCancellation()
        clear()
        let accessed = directory.startAccessingSecurityScopedResource()
        do {
            let bundle = try EmbeddingBundle(directory: directory)
            let tokenizer = try EmbeddingTokenizer(directory: bundle.directory)
            let table = try TokenEmbeddingTable(url: bundle.directory.appending(path: "token_embeddings.bf16"))
            let text = try await Self.function(bundle.directory.appending(path: "text.aimodel"), vision: false)
            try Task.checkCancellation()
            let vision = try await Self.function(bundle.directory.appending(path: "vision.aimodel"), vision: true)
            try Task.checkCancellation()
            loaded = Loaded(text: text, vision: vision, tokenizer: tokenizer, table: table)
            self.directory = accessed ? directory : nil
        } catch {
            if accessed { directory.stopAccessingSecurityScopedResource() }
            throw error
        }
    }

    public func unload() throws {
        guard !busy else { throw EmbeddingError.busy }
        clear()
    }

    public func encode(_ text: String, role: EmbeddingTextRole = .query) async throws -> EmbeddingVector {
        try begin()
        defer { busy = false }
        guard let loaded else { throw EmbeddingError.notLoaded }
        let tokens = try loaded.tokenizer.tokens(for: text, role: role)
        return try await Self.embed(loaded, tokens: tokens, embeddings: loaded.table.embeddings(for: tokens))
    }

    public func encode(imageURL: URL) async throws -> EmbeddingVector {
        try begin()
        defer { busy = false }
        guard let loaded else { throw EmbeddingError.notLoaded }
        let image = try EmbeddingImage(EmbeddingImageReader.read(imageURL))
        try Task.checkCancellation()
        let tokens = [2, 255_999] + Array(repeating: 258_880, count: image.softTokenCount) + [258_882, 1]
        var embeddings = try loaded.table.embeddings(for: tokens)
        let inputs: [String: NDArray] = [
            "pixel_values": NDArray(scalars: image.pixels.map(Float16.init), shape: [1, 630, 768]),
            "image_position_ids": NDArray(scalars: image.positions, shape: [1, 630, 2]),
            "pooling_weights": NDArray(scalars: image.pooling.map(Float16.init), shape: [1, 70, 630]),
        ]
        var outputs = try await loaded.vision.run(inputs: inputs)
        let softTokens = try Self.floats(outputs.remove("soft_tokens")?.ndArray, count: 70 * 512)
        try Task.checkCancellation()
        for token in 0..<image.softTokenCount {
            embeddings.replaceSubrange(
                (token + 2) * 512..<(token + 3) * 512,
                with: softTokens[token * 512..<(token + 1) * 512])
        }
        return try await Self.embed(loaded, tokens: tokens, embeddings: embeddings)
    }

    private func begin() throws {
        guard !busy else { throw EmbeddingError.busy }
        try Task.checkCancellation()
        busy = true
    }

    private func clear() {
        loaded = nil
        directory?.stopAccessingSecurityScopedResource()
        directory = nil
    }

    private static func embed(_ loaded: Loaded, tokens: [Int], embeddings: [Float]) async throws -> EmbeddingVector {
        try Task.checkCancellation()
        let mask = (0..<128).map { Float16($0 < tokens.count ? 1 : 0) }
        var outputs = try await loaded.text.run(inputs: [
            "inputs_embeds": NDArray(scalars: embeddings.map(Float16.init), shape: [1, 128, 512]),
            "attention_mask": NDArray(scalars: mask, shape: [1, 128]),
        ])
        let values = try floats(outputs.remove("embedding")?.ndArray, count: 768)
        try Task.checkCancellation()
        return try EmbeddingVector(values)
    }

    private static func function(_ url: URL, vision: Bool) async throws -> InferenceFunction {
        let model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        guard let descriptor = model.functionDescriptor(for: "main") else {
            throw EmbeddingError.invalidBundle("Missing main function")
        }
        let inputs: [String: [Int]] =
            vision
            ? ["pixel_values": [1, 630, 768], "image_position_ids": [1, 630, 2], "pooling_weights": [1, 70, 630]]
            : ["inputs_embeds": [1, 128, 512], "attention_mask": [1, 128]]
        let outputs = vision ? ["soft_tokens": [1, 70, 512]] : ["embedding": [1, 768]]
        guard Set(descriptor.inputNames) == Set(inputs.keys), Set(descriptor.outputNames) == Set(outputs.keys) else {
            throw EmbeddingError.invalidBundle("Incorrect function names")
        }
        for (name, shape) in inputs {
            try check(descriptor.inputDescriptor(of: name), name: name, shape: shape)
        }
        for (name, shape) in outputs {
            try check(descriptor.outputDescriptor(of: name), name: name, shape: shape)
        }
        guard let function = try model.loadFunction(named: "main") else {
            throw EmbeddingError.invalidBundle("Missing main function")
        }
        return function
    }

    private static func check(_ value: InferenceValue.Descriptor?, name: String, shape: [Int]) throws {
        let scalar: NDArray.ScalarType = name == "image_position_ids" ? .int32 : .float16
        guard case .ndArray(let array)? = value, array.shape == shape, array.scalarType == scalar else {
            throw EmbeddingError.invalidBundle("Incorrect shape or scalar type: " + name)
        }
    }

    private static func floats(_ array: NDArray?, count: Int) throws -> [Float] {
        guard let array, array.scalarType == .float16,
            let values = array.view(as: Float16.self).contiguousElements?.withUnsafeBufferPointer({ $0.map(Float.init) }
            ),
            values.count == count, values.allSatisfy(\.isFinite)
        else { throw EmbeddingError.invalidVector }
        return values
    }
}
