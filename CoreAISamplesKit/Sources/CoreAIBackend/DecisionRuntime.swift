import CoreAI
import Foundation

struct DecisionRuntime: Sendable {
    let backbone: AIModel
    let head: AIModel
    let function: InferenceFunction
    let headFunction: InferenceFunction
    let hiddenDescriptor: NDArrayDescriptor
    let logitsDescriptor: NDArrayDescriptor
    let capacity: Int
    let width: Int
    let questions: Int
    let options: Int
    var keyCache: NDArray
    var valueCache: NDArray
    var convStates: NDArray
    var recurrentStates: NDArray
    let convStateByteCount: Int
    let recurrentStateByteCount: Int
    var processed = 0

    struct RetainedStates: Sendable {
        let capacity: Int
        let keyCache: NDArray
        let valueCache: NDArray
        let convStates: NDArray
        let recurrentStates: NDArray

        func array(named name: String) -> NDArray? {
            switch name {
            case "keyCache": keyCache
            case "valueCache": valueCache
            case "convStates": convStates
            case "recurrentStates": recurrentStates
            default: nil
            }
        }
    }

    func retainedStates() -> RetainedStates {
        RetainedStates(
            capacity: capacity, keyCache: keyCache, valueCache: valueCache,
            convStates: convStates, recurrentStates: recurrentStates)
    }

    struct Checkpoint: Sendable {
        let convolution: Data
        let recurrence: Data
    }

    func checkpoint() -> Checkpoint {
        Checkpoint(
            convolution: convStates.rawView().withUnsafeBytes { pointer, _, _ in
                Data(bytes: pointer, count: convStateByteCount)
            },
            recurrence: recurrentStates.rawView().withUnsafeBytes { pointer, _, _ in
                Data(bytes: pointer, count: recurrentStateByteCount)
            })
    }

    mutating func restore(_ checkpoint: Checkpoint, processed: Int) throws {
        guard checkpoint.convolution.count == convStateByteCount,
            checkpoint.recurrence.count == recurrentStateByteCount
        else { throw DecisionError.invalidOutput("The saved decision state has the wrong size.") }
        convStates.mutableRawView().withUnsafeMutableBytes { destination, _, _ in
            checkpoint.convolution.withUnsafeBytes { source in
                if let pointer = source.baseAddress { destination.copyMemory(from: pointer, byteCount: source.count) }
            }
        }
        recurrentStates.mutableRawView().withUnsafeMutableBytes { destination, _, _ in
            checkpoint.recurrence.withUnsafeBytes { source in
                if let pointer = source.baseAddress { destination.copyMemory(from: pointer, byteCount: source.count) }
            }
        }
        self.processed = processed
    }

    static func load(
        bundle: DecisionBundle, capacity: Int, retaining retained: RetainedStates? = nil
    ) async throws -> Self {
        for (url, functionName, outputName) in [
            (bundle.backboneURL, bundle.manifest.backbone.function, "hidden_states"),
            (bundle.headURL, bundle.manifest.head.function, "logits"),
        ] {
            do {
                let summary = try AIModelAsset(contentsOf: url).summary(includingStatistics: false)
                guard
                    summary?.functions.first(where: { $0.name == functionName })?.outputs.contains(
                        where: { $0.name == outputName }) == true
                else { throw DecisionError.invalidBundle("The asset is not the declared decision model.") }
            } catch {
                throw DecisionError.invalidBundle(error.localizedDescription)
            }
        }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        func load(_ url: URL) async throws -> AIModel {
            if let cached = try AIModelCache.default.model(for: url, options: options) { return cached }
            return try await AIModel.specialize(contentsOf: url, options: options)
        }
        let backbone = try await load(bundle.backboneURL)
        try Task.checkCancellation()
        let head = try await load(bundle.headURL)
        guard let descriptor = backbone.functionDescriptor(for: bundle.manifest.backbone.function),
            let function = try backbone.loadFunction(named: bundle.manifest.backbone.function),
            let headDescriptor = head.functionDescriptor(for: bundle.manifest.head.function),
            let headFunction = try head.loadFunction(named: bundle.manifest.head.function),
            case .ndArray(let hidden) = descriptor.outputDescriptor(of: "hidden_states"),
            case .ndArray(let logits) = headDescriptor.outputDescriptor(of: "logits"),
            hidden.scalarType == .float16, hidden.shape.count == 3,
            hidden.shape[0] == 1, hidden.shape[2] == bundle.manifest.backbone.hiddenSize,
            logits.scalarType == .float16, logits.shape == [bundle.limits.questions, bundle.limits.optionsPerQuestion],
            Set(descriptor.stateNames) == Set(bundle.manifest.backbone.states),
            Set(descriptor.inputNames) == Set(bundle.manifest.backbone.inputs),
            Set(headDescriptor.inputNames) == Set(bundle.manifest.head.inputs)
        else { throw DecisionError.invalidBundle("The model descriptors do not match decision.json.") }
        let width = bundle.manifest.backbone.hiddenSize
        guard hidden.shape[1] < 0, hidden.interleaveLayout == nil, logits.interleaveLayout == nil,
            headDescriptor.stateNames.isEmpty
        else { throw DecisionError.invalidBundle("Unsupported hidden, logits or head state layout.") }
        for name in ["input_ids", "position_ids"] {
            try Self.validate(descriptor.inputDescriptor(of: name), type: .int32, shape: [1, -1], name: name)
        }
        for (name, type, shape) in [
            ("hidden", NDArray.ScalarType.float16, [1, -1, width]),
            ("valid_len", .int32, [1]),
            ("question_spans", .int32, [16, 2]),
            ("option_spans", .int32, [16, 8, 2]),
            ("question_types", .int32, [16]),
            ("lexical", .float16, [16, 8, width]),
        ] {
            try Self.validate(headDescriptor.inputDescriptor(of: name), type: type, shape: shape, name: name)
        }
        for (name, shape) in [
            ("keyCache", [8, 1, 4, -1, 256]),
            ("valueCache", [8, 1, 4, -1, 256]),
            ("convStates", [24, 1, 4, 8_192]),
            ("recurrentStates", [24, 1, 32, 128, 128]),
        ] {
            try Self.validate(descriptor.stateDescriptor(of: name), type: .bfloat16, shape: shape, name: name)
        }
        func state(_ name: String) throws -> NDArray {
            guard case .ndArray(let spec) = descriptor.stateDescriptor(of: name) else {
                throw DecisionError.invalidBundle("Missing state: \(name).")
            }
            let shape = spec.shape.map { $0 < 0 ? capacity : $0 }
            let resolved = spec.resolvingDynamicDimensions(shape)
            if let retained, let array = retained.array(named: name) {
                guard retained.capacity == capacity, array.shape == shape,
                    array.scalarType == spec.scalarType, array.strides == resolved.preferredStrides
                else { throw DecisionError.invalidBundle("The retained state layout changed.") }
                return array
            }
            var array = NDArray(descriptor: resolved)
            let bytes = resolved.minimumByteCount
            array.mutableRawView().withUnsafeMutableBytes { pointer, _, _ in
                _ = pointer.initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
            }
            return array
        }
        func fixedSize(_ name: String) throws -> Int {
            guard case .ndArray(let array) = descriptor.stateDescriptor(of: name) else {
                throw DecisionError.invalidBundle("Missing fixed state.")
            }
            return array.minimumByteCount
        }
        return try Self(
            backbone: backbone, head: head, function: function, headFunction: headFunction,
            hiddenDescriptor: hidden, logitsDescriptor: logits, capacity: capacity,
            width: hidden.shape[2], questions: logits.shape[0], options: logits.shape[1],
            keyCache: state("keyCache"), valueCache: state("valueCache"),
            convStates: state("convStates"), recurrentStates: state("recurrentStates"),
            convStateByteCount: fixedSize("convStates"), recurrentStateByteCount: fixedSize("recurrentStates"))
    }

    mutating func forward(_ ids: ArraySlice<Int>) async throws -> [Float16] {
        try Task.checkCancellation()
        let end = processed + ids.count
        guard !ids.isEmpty, end <= capacity,
            ids.allSatisfy({ $0 >= 0 && $0 <= Int32.max })
        else { throw DecisionError.invalidRequest("Invalid backbone input length or token.") }
        let inputs = [
            "input_ids": Self.integers(ids.map(Int32.init), shape: [1, ids.count]),
            "position_ids": Self.integers((0..<end).map(Int32.init), shape: [1, end]),
        ]
        var hidden = NDArray(descriptor: hiddenDescriptor.resolvingDynamicDimensions([1, ids.count, width]))
        let function = function
        var stateViews = InferenceFunction.MutableViews()
        stateViews.insert(&keyCache, for: "keyCache")
        stateViews.insert(&valueCache, for: "valueCache")
        stateViews.insert(&convStates, for: "convStates")
        stateViews.insert(&recurrentStates, for: "recurrentStates")
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&hidden, for: "hidden_states")
        _ = try await function.run(inputs: inputs, states: stateViews, outputViews: outputViews)
        processed = end
        try Task.checkCancellation()
        let rows = ids.count
        var output = [Float16](repeating: 0, count: rows * width)
        hidden.view(as: Float16.self).withUnsafePointer { pointer, _, strides in
            output.withUnsafeMutableBufferPointer { destination in
                guard let target = destination.baseAddress else { return }
                for row in 0..<rows {
                    if strides[2] == 1 {
                        target.advanced(by: row * width).update(
                            from: pointer.advanced(by: row * strides[1]), count: width)
                    } else {
                        for column in 0..<width {
                            target[row * width + column] = pointer[row * strides[1] + column * strides[2]]
                        }
                    }
                }
            }
        }
        return output
    }

    func predict(hidden: [Float16], rest: [Float16] = [], rendered: RenderedDecision, lexical: [Float16]) async throws
        -> [Float]
    {
        try Task.checkCancellation()
        let tokens = rendered.tokenIDs.count
        let padded = (tokens + 511) / 512 * 512
        guard hidden.count + rest.count == tokens * width,
            lexical.count == questions * options * width
        else { throw DecisionError.invalidOutput("Hidden or lexical tensor dimensions do not match.") }
        var input = NDArray(shape: [1, padded, width], scalarType: .float16)
        input.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, strides in
            let first = hidden.count / width
            for row in 0..<padded {
                for column in 0..<width {
                    pointer[row * strides[1] + column * strides[2]] =
                        row < first
                        ? hidden[row * width + column]
                        : row < tokens ? rest[(row - first) * width + column] : 0
                }
            }
        }
        var qspan = [Int32](repeating: 0, count: questions * 2)
        var ospan = [Int32](repeating: 0, count: questions * options * 2)
        var qtype = [Int32](repeating: 0, count: questions)
        for index in rendered.questionSpans.indices {
            let span = rendered.questionSpans[index]
            qspan[index * 2] = Int32(span.lowerBound)
            qspan[index * 2 + 1] = Int32(span.upperBound)
            qtype[index] = Int32(rendered.questionTypes[index])
            for (option, span) in rendered.optionSpans[index].enumerated() {
                let slot = (index * options + option) * 2
                ospan[slot] = Int32(span.lowerBound)
                ospan[slot + 1] = Int32(span.upperBound)
            }
        }
        var lex = NDArray(shape: [questions, options, width], scalarType: .float16)
        lex.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, _ in
            for index in lexical.indices { pointer[index] = lexical[index] }
        }
        let inputs = [
            "hidden": input,
            "valid_len": Self.integers([Int32(tokens)], shape: [1]),
            "question_spans": Self.integers(qspan, shape: [questions, 2]),
            "option_spans": Self.integers(ospan, shape: [questions, options, 2]),
            "question_types": Self.integers(qtype, shape: [questions]),
            "lexical": lex,
        ]
        var logits = NDArray(descriptor: logitsDescriptor)
        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&logits, for: "logits")
        _ = try await headFunction.run(inputs: inputs, outputViews: outputs)
        try Task.checkCancellation()
        var values = [Float](repeating: 0, count: questions * options)
        logits.view(as: Float16.self).withUnsafePointer { pointer, _, strides in
            for question in 0..<questions {
                for option in 0..<options {
                    values[question * options + option] = Float(pointer[question * strides[0] + option * strides[1]])
                }
            }
        }
        return values
    }

    private static func validate(
        _ descriptor: InferenceValue.Descriptor?, type: NDArray.ScalarType, shape: [Int], name: String
    ) throws {
        guard case .ndArray(let array) = descriptor, array.scalarType == type,
            array.interleaveLayout == nil, array.shape.count == shape.count,
            zip(array.shape, shape).allSatisfy({ $1 < 0 ? $0 < 0 : $0 == $1 })
        else { throw DecisionError.invalidBundle("Unsupported tensor descriptor: \(name).") }
    }

    private static func integers(_ values: [Int32], shape: [Int]) -> NDArray {
        var array = NDArray(shape: shape, scalarType: .int32)
        array.mutableView(as: Int32.self).withUnsafeMutablePointer { pointer, _, _ in
            for index in values.indices { pointer[index] = values[index] }
        }
        return array
    }
}
