#if os(macOS) || os(iOS)
    import CoreAI
    import Foundation
    import SpeechCore

    struct Stage: Sendable {
        let model: AIModel
        let function: InferenceFunction
        private let assetName: String
        private let tracing: Bool
        init(url: URL, dynamicShape: Bool, frequentReshapes: Bool? = nil) async throws {
            try Task.checkCancellation()
            let name = url.lastPathComponent
            let trace = ProcessInfo.processInfo.environment["COREAI_TRACE_STAGES"] == "1"
            assetName = name
            tracing = trace
            #if os(iOS)
                let onPhone = true
            #else
                let onPhone = false
            #endif
            var options = Self.options(dynamicShape: dynamicShape, onPhone: onPhone)
            if let frequentReshapes { options.expectFrequentReshapes = frequentReshapes }
            if trace { print("Irodori specialize \(name), frequentReshapes=\(options.expectFrequentReshapes)") }
            model = try await AIModel.specialize(contentsOf: url, options: options)
            try Task.checkCancellation()
            guard let descriptor = model.functionDescriptor(for: "predict"), descriptor.stateNames.isEmpty,
                case .ndArray(let output) = descriptor.outputDescriptor(of: "output"), output.scalarType == .float32,
                let function = try model.loadFunction(named: "predict")
            else { throw SpeechError.invalidAsset("Unsupported predict/output/state contract") }
            self.function = function
            if trace { print("Irodori ready \(name)") }
        }
        static func options(dynamicShape: Bool, onPhone: Bool) -> SpecializationOptions {
            var result = SpecializationOptions(preferredComputeUnitKind: .gpu)
            result.expectFrequentReshapes = !onPhone || dynamicShape
            return result
        }
        @concurrent
        func run(_ inputs: [String: NDArray]) async throws -> NDArray {
            try Task.checkCancellation()
            if tracing { print("Irodori run \(assetName): \(inputs.mapValues { $0.shape })") }
            var outputs = try await function.run(inputs: inputs)
            guard let result = outputs.remove("output"), let array = result.ndArray else {
                throw SpeechError.invalidAsset("Missing output")
            }
            try Task.checkCancellation()
            if tracing { print("Irodori output \(assetName): \(array.shape)") }
            return array
        }
    }

    func floats(_ array: NDArray) throws -> [Float] {
        guard array.scalarType == .float32, array.interleaveLayout == nil else { throw SpeechError.invalidSamples }
        let shape = array.shape
        guard shape.allSatisfy({ $0 > 0 }) else { throw SpeechError.invalidSamples }
        var count = 1
        for dimension in shape {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard !product.overflow, product.partialValue <= 750 * 1920 else { throw SpeechError.invalidSamples }
            count = product.partialValue
        }
        return array.view(as: Float.self).withUnsafePointer { pointer, _, strides in
            (0..<count).map { index in
                var remaining = index
                var offset = 0
                for dimension in shape.indices.reversed() {
                    offset += (remaining % shape[dimension]) * strides[dimension]
                    remaining /= shape[dimension]
                }
                return pointer[offset]
            }
        }
    }

#endif
