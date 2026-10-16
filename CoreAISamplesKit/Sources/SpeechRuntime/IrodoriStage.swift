#if os(macOS) || os(iOS)
    import CoreAI
    import Foundation
    import SpeechCore

    struct Stage: Sendable {
        let model: AIModel
        let function: InferenceFunction
        init(url: URL, dynamicShape: Bool) async throws {
            try Task.checkCancellation()
            #if os(iOS)
                let onPhone = true
            #else
                let onPhone = false
            #endif
            let options = Self.options(dynamicShape: dynamicShape, onPhone: onPhone)
            model = try await AIModel.specialize(contentsOf: url, options: options)
            try Task.checkCancellation()
            guard let descriptor = model.functionDescriptor(for: "predict"), descriptor.stateNames.isEmpty,
                case .ndArray(let output) = descriptor.outputDescriptor(of: "output"), output.scalarType == .float32,
                let function = try model.loadFunction(named: "predict")
            else { throw SpeechError.invalidAsset("Unsupported predict/output/state contract") }
            self.function = function
        }
        static func options(dynamicShape: Bool, onPhone: Bool) -> SpecializationOptions {
            var result = SpecializationOptions(preferredComputeUnitKind: .gpu)
            result.expectFrequentReshapes = !onPhone || dynamicShape
            return result
        }
        @concurrent
        func run(_ inputs: [String: NDArray]) async throws -> NDArray {
            try Task.checkCancellation()
            var outputs = try await function.run(inputs: inputs)
            guard let result = outputs.remove("output"), let array = result.ndArray else {
                throw SpeechError.invalidAsset("Missing output")
            }
            try Task.checkCancellation()
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
