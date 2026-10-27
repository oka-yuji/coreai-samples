import CoreAI
import Foundation
import OCRCore
import Tokenizers

struct CoreAIOCRModel: OCRModel {
    struct Cache: Sendable {
        var keyCache: NDArray
        var valueCache: NDArray
    }

    private struct Metadata: Decodable {
        struct Assets: Decodable {
            let vision: String
            let embed: String
            let main: String
        }

        struct Vision: Decodable { let windowLimit: Int }
        struct Text: Decodable { let tokenLimit: Int }

        let format: String
        let assets: Assets
        let tokenizer: String
        let vision: Vision
        let text: Text
    }

    private static let signatures:
        [(
            asset: String, function: String, inputs: [String: NDArray.ScalarType],
            output: (String, NDArray.ScalarType), states: [String]
        )] = [
            (
                "vision", "main", ["pixel_values": .float16, "key_valid": .float16, "cos": .float16, "sin": .float16],
                ("image_features", .float16), []
            ),
            ("embed", "main", ["input_ids": .int32], ("inputs_embeds", .float32), []),
            (
                "main", "prefill", ["inputs_embeds": .float16, "cos": .float16, "sin": .float16, "mask": .float16],
                ("logits", .float16), ["keyCache", "valueCache"]
            ),
            (
                "main", "main", ["input_ids": .int32, "cos": .float16, "sin": .float16, "mask": .float16],
                ("logits", .float16), ["keyCache", "valueCache"]
            ),
        ]

    let tokenizer: any Tokenizer
    let windowLimit: Int
    let tokenLimit: Int
    let models: [AIModel]
    let functions: [String: InferenceFunction]
    let states: [String: NDArrayDescriptor]

    static func load(bundle: URL) async throws -> (CoreAIOCRModel, OCRLoadStats) {
        let clock = ContinuousClock()
        let start = clock.now
        let metadata = try Self.metadata(bundle)
        let tokenizer = try await AutoTokenizer.from(modelFolder: bundle.appending(path: metadata.tokenizer))
        let tokenizerSeconds = (clock.now - start) / .seconds(1)
        var options = SpecializationOptions.default
        options.expectFrequentReshapes = true
        var models: [String: AIModel] = [:]
        var assets: [OCRLoadStats.Asset] = []
        for (name, folder) in [
            ("vision", metadata.assets.vision), ("embed", metadata.assets.embed), ("main", metadata.assets.main),
        ] {
            try Task.checkCancellation()
            let assetStart = clock.now
            let url = bundle.appending(path: folder)
            let cached = try AIModelCache.default.model(for: url, options: options)
            let model: AIModel
            if let cached {
                model = cached
            } else {
                model = try await AIModel.specialize(contentsOf: url, options: options)
            }
            models[name] = model
            assets.append(.init(name: name, cached: cached != nil, seconds: (clock.now - assetStart) / .seconds(1)))
        }
        let (functions, states) = try bind(models)
        let model = CoreAIOCRModel(
            tokenizer: tokenizer, windowLimit: metadata.vision.windowLimit, tokenLimit: metadata.text.tokenLimit,
            models: Array(models.values), functions: functions, states: states)
        return (
            model,
            OCRLoadStats(
                assets: assets, tokenizerSeconds: tokenizerSeconds, totalSeconds: (clock.now - start) / .seconds(1))
        )
    }

    private static func bind(_ models: [String: AIModel]) throws
        -> (functions: [String: InferenceFunction], states: [String: NDArrayDescriptor])
    {
        var functions: [String: InferenceFunction] = [:]
        var states: [String: NDArrayDescriptor] = [:]
        for signature in signatures {
            guard let model = models[signature.asset],
                let descriptor = model.functionDescriptor(for: signature.function),
                Set(descriptor.inputNames) == Set(signature.inputs.keys),
                descriptor.outputNames == [signature.output.0], Set(descriptor.stateNames) == Set(signature.states),
                signature.inputs.allSatisfy({ Self.type(descriptor.inputDescriptor(of: $0.key)) == $0.value }),
                Self.type(descriptor.outputDescriptor(of: signature.output.0)) == signature.output.1,
                let function = try model.loadFunction(named: signature.function)
            else {
                throw OCRRuntimeError.invalidBundle("\(signature.asset).\(signature.function) breaks the contract.")
            }
            for name in signature.states {
                guard case .ndArray(let state) = descriptor.stateDescriptor(of: name), state.scalarType == .float16,
                    state.shape.filter({ $0 < 0 }).count == 1
                else { throw OCRRuntimeError.invalidBundle("Unsupported state \(name).") }
                states[name] = state
            }
            functions[signature.asset + "." + signature.function] = function
        }
        return (functions, states)
    }

    static func deleteSpecializationCache(bundle: URL) throws {
        let assets = try metadata(bundle).assets
        for folder in [assets.vision, assets.embed, assets.main] {
            try AIModelCache.default.deleteEntries(for: bundle.appending(path: folder))
        }
    }

    private static func metadata(_ bundle: URL) throws -> Metadata {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let metadata: Metadata
        do {
            metadata = try decoder.decode(
                Metadata.self, from: Data(contentsOf: bundle.appending(path: "metadata.json")))
        } catch {
            throw OCRRuntimeError.invalidBundle("metadata.json: \(error.localizedDescription)")
        }
        guard metadata.format == "coreai-samples-teleocr/1" else {
            throw OCRRuntimeError.invalidBundle("Unsupported format \(metadata.format).")
        }
        return metadata
    }

    private static func type(_ descriptor: InferenceValue.Descriptor?) -> NDArray.ScalarType? {
        guard case .ndArray(let array) = descriptor, array.interleaveLayout == nil else { return nil }
        return array.scalarType
    }

    func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map { Int32($0) }
    }

    func decode(_ tokens: [Int32], skipSpecialTokens: Bool) -> String {
        tokenizer.decode(tokens: tokens.map { Int($0) }, skipSpecialTokens: skipSpecialTokens)
    }

    func makeCache(capacity: Int) throws -> Cache {
        func state(_ name: String) throws -> NDArray {
            guard let descriptor = states[name] else { throw OCRRuntimeError.invalidBundle("Missing state \(name).") }
            return NDArray(
                descriptor: descriptor.resolvingDynamicDimensions(descriptor.shape.map { $0 < 0 ? capacity : $0 }))
        }
        return Cache(keyCache: try state("keyCache"), valueCache: try state("valueCache"))
    }

    @concurrent func vision(_ input: OCRVisionInput) async throws -> [Float] {
        let rows = input.rowCount
        var outputs = try await function("vision.main").run(inputs: [
            "pixel_values": Self.half(input.pixelValues, [rows, OCRVisionInput.rowWidth]),
            "key_valid": Self.half(input.keyValid, [rows]),
            "cos": Self.half(input.cos, [rows, OCRVisionInput.rotaryWidth]),
            "sin": Self.half(input.sin, [rows, OCRVisionInput.rotaryWidth]),
        ])
        return try Self.floats(outputs.remove("image_features")?.ndArray)
    }

    @concurrent func embed(_ ids: [Int32]) async throws -> [Float] {
        var outputs = try await function("embed.main").run(inputs: [
            "input_ids": NDArray(scalars: ids, shape: [1, ids.count])
        ])
        return try Self.floats(outputs.remove("inputs_embeds")?.ndArray)
    }

    @concurrent func prefill(_ embeddings: [Float], rotary: OCRRotary, cache: inout Cache) async throws -> [Float] {
        let rows = rotary.cos.count / OCRTextRotary.width
        let inputs = [
            "inputs_embeds": Self.half(embeddings, [1, rows, embeddings.count / rows]),
            "cos": Self.half(rotary.cos, [1, rows, OCRTextRotary.width]),
            "sin": Self.half(rotary.sin, [1, rows, OCRTextRotary.width]),
            "mask": Self.mask(rows: rows, position: 0),
        ]
        return try await Self.forward(try function("main.prefill"), inputs, cache: &cache)
    }

    @concurrent func step(_ token: Int32, position: Int, rotary: OCRRotary, cache: inout Cache) async throws
        -> [Float]
    {
        let inputs = [
            "input_ids": NDArray(scalars: [token], shape: [1, 1]),
            "cos": Self.half(rotary.cos, [1, 1, OCRTextRotary.width]),
            "sin": Self.half(rotary.sin, [1, 1, OCRTextRotary.width]),
            "mask": Self.mask(rows: 1, position: position),
        ]
        return try await Self.forward(try function("main.main"), inputs, cache: &cache)
    }

    @concurrent private static func forward(
        _ function: InferenceFunction, _ inputs: [String: NDArray], cache: inout Cache
    )
        async throws -> [Float]
    {
        var states = InferenceFunction.MutableViews()
        states.insert(&cache.keyCache, for: "keyCache")
        states.insert(&cache.valueCache, for: "valueCache")
        var outputs = try await function.run(inputs: inputs, states: states)
        return try floats(outputs.remove("logits")?.ndArray)
    }

    private func function(_ name: String) throws -> InferenceFunction {
        guard let function = functions[name] else { throw OCRRuntimeError.invalidBundle("Missing function \(name).") }
        return function
    }

    static func mask(rows: Int, position: Int) -> NDArray {
        let columns = position + rows
        var values = [Float16](repeating: 0, count: rows * columns)
        for row in 0..<rows {
            for column in (position + row + 1)..<columns { values[row * columns + column] = -.infinity }
        }
        return NDArray(scalars: values, shape: [1, 1, rows, columns])
    }

    static func half(_ values: [Float], _ shape: [Int]) -> NDArray {
        NDArray(scalars: values.lazy.map { Float16($0) }, shape: shape)
    }

    static func floats(_ array: NDArray?) throws -> [Float] {
        guard let array, array.interleaveLayout == nil else { throw OCRRuntimeError.invalidOutput("Missing output.") }
        let shape = array.shape
        let strides = array.strides
        var contiguous = [Int](repeating: 1, count: shape.count)
        for axis in shape.indices.dropLast().reversed() { contiguous[axis] = contiguous[axis + 1] * shape[axis + 1] }
        let dense = strides == contiguous
        func offset(_ index: Int) -> Int {
            var rest = index
            var offset = 0
            for axis in shape.indices.reversed() {
                offset += rest % shape[axis] * strides[axis]
                rest /= shape[axis]
            }
            return offset
        }
        let count = shape.reduce(1, *)
        switch array.scalarType {
        case .float16:
            return array.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
                (0..<count).map { Float(pointer[dense ? $0 : offset($0)]) }
            }
        case .float32:
            return array.view(as: Float.self).withUnsafePointer { pointer, _, _ in
                (0..<count).map { pointer[dense ? $0 : offset($0)] }
            }
        default:
            throw OCRRuntimeError.invalidOutput("Unsupported output type \(array.scalarType).")
        }
    }
}
