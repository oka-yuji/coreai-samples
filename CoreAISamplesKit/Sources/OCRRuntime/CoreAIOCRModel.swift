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
        struct Static: Decodable {
            let context: Int
            let masked: Float
            let visionWindows: [Int]
            let downscaleSide: Int
            let embedTokens: Int
            let prefillTokens: [Int]
        }

        let format: String
        let platform: String?
        let assets: Assets
        let tokenizer: String
        let vision: Vision
        let text: Text
        let `static`: Static?
    }

    private typealias Signature = (
        asset: String, function: String, inputs: [String: NDArray.ScalarType], output: (String, NDArray.ScalarType),
        states: [String]
    )

    private static let signatures: [Signature] = [
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
    let staticShapes: OCRStaticShapes?
    let models: [AIModel]
    let functions: [String: InferenceFunction]
    let states: [String: NDArrayDescriptor]

    static func load(bundle: URL) async throws -> (CoreAIOCRModel, OCRLoadStats) {
        let clock = ContinuousClock()
        let start = clock.now
        let metadata = try Self.metadata(bundle)
        let shapes = try Self.shapes(metadata)
        let tokenizer = try await AutoTokenizer.from(modelFolder: bundle.appending(path: metadata.tokenizer))
        let tokenizerSeconds = (clock.now - start) / .seconds(1)
        var options = SpecializationOptions.default
        options.expectFrequentReshapes = shapes == nil
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
        let (functions, states) = try bind(models, shapes: shapes)
        let model = CoreAIOCRModel(
            tokenizer: tokenizer, windowLimit: metadata.vision.windowLimit, tokenLimit: metadata.text.tokenLimit,
            staticShapes: shapes, models: Array(models.values), functions: functions, states: states)
        return (
            model,
            OCRLoadStats(
                assets: assets, tokenizerSeconds: tokenizerSeconds, totalSeconds: (clock.now - start) / .seconds(1))
        )
    }

    private static func staticSignatures(_ shapes: OCRStaticShapes) -> [Signature] {
        let half = NDArray.ScalarType.float16
        let caches = ["keyCache", "valueCache"]
        let vision: [Signature] = shapes.visionWindows.map {
            (
                "vision", "windows_\($0)", ["pixel_values": half, "key_valid": half, "cos": half, "sin": half],
                ("image_features", half), []
            )
        }
        let prefill: [Signature] = shapes.prefillTokens.map {
            (
                "main", "prefill_\($0)",
                ["inputs_embeds": half, "cos": half, "sin": half, "mask": half, "logit_row": .int32], ("logits", half),
                caches
            )
        }
        return vision + [("embed", "main", ["input_ids": .int32], ("inputs_embeds", .float32), [])] + prefill + [
            (
                "main", "main", ["input_ids": .int32, "cos": half, "sin": half, "mask": half, "write_mask": half],
                ("logits", half), caches
            )
        ]
    }

    private static func bind(_ models: [String: AIModel], shapes: OCRStaticShapes?) throws
        -> (functions: [String: InferenceFunction], states: [String: NDArrayDescriptor])
    {
        var functions: [String: InferenceFunction] = [:]
        var states: [String: NDArrayDescriptor] = [:]
        for signature in shapes.map(staticSignatures) ?? signatures {
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
                    shapes.map({
                        !state.hasDynamicShape && state.shape.count == 5 && state.shape[3] == $0.context
                            && (states[name] ?? state) == state
                    }) ?? (state.shape.filter({ $0 < 0 }).count == 1)
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

    private static func shapes(_ metadata: Metadata) throws -> OCRStaticShapes? {
        switch (metadata.platform, metadata.static) {
        case (nil, nil), ("macos"?, nil):
            return nil
        case ("ios"?, let shapes?):
            let shapes = OCRStaticShapes(
                context: shapes.context, masked: shapes.masked, visionWindows: shapes.visionWindows,
                downscaleSide: shapes.downscaleSide, embedTokens: shapes.embedTokens,
                prefillTokens: shapes.prefillTokens)
            guard let windows = shapes.visionWindows.last, windows == metadata.vision.windowLimit,
                shapes.context == metadata.text.tokenLimit, let tokens = shapes.prefillTokens.first,
                tokens > 0, shapes.prefillTokens.last == shapes.embedTokens, shapes.embedTokens < shapes.context,
                shapes.visionWindows[0] > 0, shapes.masked.isFinite, shapes.masked < 0
            else { throw OCRRuntimeError.invalidBundle("Inconsistent static shapes.") }
            return shapes
        default:
            throw OCRRuntimeError.invalidBundle("Unsupported platform layout.")
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
            if let shapes = staticShapes {
                guard capacity <= shapes.context else {
                    throw OCRRuntimeError.invalidRequest("\(capacity) tokens exceed the \(shapes.context) context.")
                }
                var array = NDArray(descriptor: descriptor)
                let count = descriptor.minimumByteCount
                array.mutableRawView().withUnsafeMutableBytes { pointer, _, _ in
                    _ = pointer.initializeMemory(as: UInt8.self, repeating: 0, count: count)
                }
                return array
            }
            return NDArray(
                descriptor: descriptor.resolvingDynamicDimensions(descriptor.shape.map { $0 < 0 ? capacity : $0 }))
        }
        return Cache(keyCache: try state("keyCache"), valueCache: try state("valueCache"))
    }

    @concurrent func vision(_ input: OCRVisionInput) async throws -> [Float] {
        if let shapes = staticShapes { return try await staticVision(input, shapes) }
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
        if let shapes = staticShapes { return try await staticEmbed(ids, shapes) }
        var outputs = try await function("embed.main").run(inputs: [
            "input_ids": NDArray(scalars: ids, shape: [1, ids.count])
        ])
        return try Self.floats(outputs.remove("inputs_embeds")?.ndArray)
    }

    @concurrent func prefill(_ embeddings: [Float], rotary: OCRRotary, cache: inout Cache) async throws -> [Float] {
        if let shapes = staticShapes {
            return try await staticPrefill(embeddings, rotary: rotary, shapes, cache: &cache)
        }
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
        if let shapes = staticShapes {
            return try await staticStep(token, position: position, rotary: rotary, shapes, cache: &cache)
        }
        let inputs = [
            "input_ids": NDArray(scalars: [token], shape: [1, 1]),
            "cos": Self.half(rotary.cos, [1, 1, OCRTextRotary.width]),
            "sin": Self.half(rotary.sin, [1, 1, OCRTextRotary.width]),
            "mask": Self.mask(rows: 1, position: position),
        ]
        return try await Self.forward(try function("main.main"), inputs, cache: &cache)
    }

    @concurrent private func staticVision(_ input: OCRVisionInput, _ shapes: OCRStaticShapes) async throws -> [Float] {
        guard let windows = shapes.visionBucket(windows: input.windowCount) else {
            throw OCRRuntimeError.invalidRequest("\(input.windowCount) windows exceed the vision buckets.")
        }
        let rows = windows * OCRVisionInput.rowsPerWindow
        let width = OCRVisionInput.rotaryWidth
        var outputs = try await function("vision.windows_\(windows)").run(inputs: [
            "pixel_values": Self.half(
                OCRStaticShapes.padded(input.pixelValues, to: rows * OCRVisionInput.rowWidth, with: 0),
                [rows, OCRVisionInput.rowWidth]),
            "key_valid": Self.half(OCRStaticShapes.padded(input.keyValid, to: rows, with: 0), [rows]),
            "cos": Self.half(OCRStaticShapes.padded(input.cos, to: rows * width, with: 1), [rows, width]),
            "sin": Self.half(OCRStaticShapes.padded(input.sin, to: rows * width, with: 0), [rows, width]),
        ])
        let output = try Self.floats(outputs.remove("image_features")?.ndArray)
        return Array(output.prefix(output.count / windows * input.windowCount))
    }

    @concurrent private func staticEmbed(_ ids: [Int32], _ shapes: OCRStaticShapes) async throws -> [Float] {
        guard !ids.isEmpty, ids.count <= shapes.embedTokens else {
            throw OCRRuntimeError.invalidRequest("\(ids.count) tokens exceed the embedding input.")
        }
        let padded = ids + [Int32](repeating: 0, count: shapes.embedTokens - ids.count)
        var outputs = try await function("embed.main").run(inputs: [
            "input_ids": NDArray(scalars: padded, shape: [1, shapes.embedTokens])
        ])
        let output = try Self.floats(outputs.remove("inputs_embeds")?.ndArray)
        return Array(output.prefix(output.count / shapes.embedTokens * ids.count))
    }

    @concurrent private func staticPrefill(
        _ embeddings: [Float], rotary: OCRRotary, _ shapes: OCRStaticShapes, cache: inout Cache
    ) async throws -> [Float] {
        let width = OCRTextRotary.width
        let rows = rotary.cos.count / width
        guard rows > 0, let tokens = shapes.prefillBucket(tokens: rows) else {
            throw OCRRuntimeError.invalidRequest("\(rows) prompt tokens exceed the prefill buckets.")
        }
        let hidden = embeddings.count / rows
        let inputs = [
            "inputs_embeds": Self.half(
                OCRStaticShapes.padded(embeddings, to: tokens * hidden, with: 0), [1, tokens, hidden]),
            "cos": Self.half(OCRStaticShapes.padded(rotary.cos, to: tokens * width, with: 1), [1, tokens, width]),
            "sin": Self.half(OCRStaticShapes.padded(rotary.sin, to: tokens * width, with: 0), [1, tokens, width]),
            "mask": Self.half(shapes.prefillMask(rows: rows, bucket: tokens), [1, 1, tokens, tokens]),
            "logit_row": NDArray(scalars: [Int32(rows - 1)], shape: [1]),
        ]
        return try await Self.forward(try function("main.prefill_\(tokens)"), inputs, cache: &cache)
    }

    @concurrent private func staticStep(
        _ token: Int32, position: Int, rotary: OCRRotary, _ shapes: OCRStaticShapes, cache: inout Cache
    ) async throws -> [Float] {
        guard position < shapes.context else {
            throw OCRRuntimeError.invalidRequest("Position \(position) is outside the \(shapes.context) context.")
        }
        let inputs = [
            "input_ids": NDArray(scalars: [token], shape: [1, 1]),
            "cos": Self.half(rotary.cos, [1, 1, OCRTextRotary.width]),
            "sin": Self.half(rotary.sin, [1, 1, OCRTextRotary.width]),
            "mask": Self.half(shapes.decodeMask(position: position), [1, 1, 1, shapes.context]),
            "write_mask": Self.half(shapes.writeMask(position: position), [shapes.context]),
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
