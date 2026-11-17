#if os(macOS) || os(iOS)
    import CoreAI
    import Foundation
    import SpeechAssets
    import SpeechCore

    struct VoxCPMAssets: VoxCPMComponents {
        let models: [AIModel]
        let functions: [VoxCPMFunction: InferenceFunction]
        let signatures: [VoxCPMFunction: VoxCPMSignature]
        let states: [String: NDArrayDescriptor]

        static var options: SpecializationOptions { SpecializationOptions(preferredComputeUnitKind: .gpu) }

        static func load(assets urls: [URL], contract: VoxCPMContract) async throws -> (
            assets: Self, seconds: [String: Double]
        ) {
            guard urls.count == VoxCPMBundleInventory.assetNames.count else {
                throw SpeechError.invalidAsset("The VoxCPM2 bundle does not list its six assets.")
            }
            let clock = ContinuousClock()
            var models: [AIModel] = []
            var functions: [VoxCPMFunction: InferenceFunction] = [:]
            var signatures: [VoxCPMFunction: VoxCPMSignature] = [:]
            var states: [String: NDArrayDescriptor] = [:]
            var seconds: [String: Double] = [:]
            for (asset, url) in zip(VoxCPMBundleInventory.assetNames, urls) {
                try Task.checkCancellation()
                let start = clock.now
                let model = try await AIModel.specialize(contentsOf: url, options: options)
                var observed: [String: VoxCPMSignature] = [:]
                for name in model.functionNames {
                    guard let descriptor = model.functionDescriptor(for: name) else { throw missing(asset, name) }
                    observed[name] = VoxCPMSignature(descriptor)
                }
                try contract.check(asset: asset, functionNames: model.functionNames, observed: observed)
                let kinds = VoxCPMFunction.allCases.filter { $0.asset == asset }
                var layouts: [String: [String: NDArrayDescriptor]] = [:]
                for kind in kinds {
                    guard let descriptor = model.functionDescriptor(for: kind.name) else {
                        throw missing(asset, kind.name)
                    }
                    for name in descriptor.stateNames {
                        guard case .ndArray(let state) = descriptor.stateDescriptor(of: name) else {
                            throw missing(asset, name)
                        }
                        layouts[kind.name, default: [:]][name] = state
                    }
                }
                try VoxCPMContract.checkSharedStates(asset: asset, layouts)
                for kind in kinds {
                    guard let function = try model.loadFunction(named: kind.name) else {
                        throw missing(asset, kind.name)
                    }
                    for (name, state) in layouts[kind.name] ?? [:] { states[asset + "." + name] = state }
                    functions[kind] = function
                    signatures[kind] = observed[kind.name]
                }
                models.append(model)
                seconds[asset] = (clock.now - start) / .seconds(1)
            }
            return (Self(models: models, functions: functions, signatures: signatures, states: states), seconds)
        }

        func makeStates() throws -> VoxCPMAssetStates {
            func zeroed(_ name: String) throws -> NDArray {
                guard let descriptor = states[name] else { throw Self.missing("state", name) }
                var array = NDArray(descriptor: descriptor)
                let count = descriptor.minimumByteCount
                array.mutableRawView().withUnsafeMutableBytes { pointer, _, _ in
                    _ = pointer.initializeMemory(as: UInt8.self, repeating: 0, count: count)
                }
                return array
            }
            let caches = VoxCPMContract.cacheStateNames
            return VoxCPMAssetStates(
                textKey: try zeroed("tslm." + caches[0]), textValue: try zeroed("tslm." + caches[1]),
                residualKey: try zeroed("ralm." + caches[0]), residualValue: try zeroed("ralm." + caches[1]),
                decoder: try VoxCPMDecoderStates { try zeroed("vae." + $0) })
        }

        @concurrent
        func call(_ function: VoxCPMFunction, _ inputs: [String: VoxCPMInput], states: inout VoxCPMAssetStates)
            async throws -> [String: [Float]]
        {
            guard let loaded = functions[function], let signature = signatures[function],
                inputs.count == signature.inputs.count
            else { throw VoxCPMGenerationError.invalidInput(function.rawValue) }
            var arrays: [String: NDArray] = [:]
            for tensor in signature.inputs {
                arrays[tensor.name] = try Self.array(inputs[tensor.name], tensor, function)
            }
            var outputs: InferenceFunction.Outputs
            switch function {
            case .prefill, .step:
                outputs = try await Self.run(loaded, arrays, key: &states.textKey, value: &states.textValue)
            case .residualPrefill, .residualStep:
                outputs = try await Self.run(loaded, arrays, key: &states.residualKey, value: &states.residualValue)
            case .decode:
                outputs = try await VoxCPMDecoderStates.run(loaded, arrays, &states.decoder)
            default:
                outputs = try await loaded.run(inputs: arrays)
            }
            var result: [String: [Float]] = [:]
            for tensor in signature.outputs {
                guard let array = outputs.remove(tensor.name)?.ndArray else {
                    throw VoxCPMGenerationError.invalidOutput(function.rawValue + "." + tensor.name)
                }
                result[tensor.name] = try Self.floats(array, tensor, function)
            }
            return result
        }

        @concurrent
        private static func run(
            _ function: InferenceFunction, _ inputs: [String: NDArray], key: inout NDArray, value: inout NDArray
        ) async throws -> InferenceFunction.Outputs {
            var views = InferenceFunction.MutableViews()
            views.insert(&key, for: VoxCPMContract.cacheStateNames[0])
            views.insert(&value, for: VoxCPMContract.cacheStateNames[1])
            return try await function.run(inputs: inputs, states: views)
        }

        private static func array(_ input: VoxCPMInput?, _ tensor: VoxCPMSignature.Tensor, _ function: VoxCPMFunction)
            throws -> NDArray
        {
            let label = function.rawValue + "." + tensor.name
            let count = tensor.shape.reduce(1, *)
            switch (input, tensor.type) {
            case (.floats(let values)?, "float32") where values.count == count:
                guard values.allSatisfy(\.isFinite) else { throw VoxCPMGenerationError.nonFiniteValue(label) }
                var array = NDArray(shape: tensor.shape, scalarType: .float32)
                array.mutableView(as: Float.self).withUnsafeMutablePointer { pointer, _, _ in
                    for index in values.indices { pointer[index] = values[index] }
                }
                return array
            case (.floats(let values)?, "float16") where values.count == count:
                let halves = values.map { Float16($0) }
                guard halves.allSatisfy(\.isFinite) else { throw VoxCPMGenerationError.nonFiniteValue(label) }
                var array = NDArray(shape: tensor.shape, scalarType: .float16)
                array.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, _ in
                    for index in halves.indices { pointer[index] = halves[index] }
                }
                return array
            case (.integers(let values)?, "int32") where values.count == count:
                var array = NDArray(shape: tensor.shape, scalarType: .int32)
                array.mutableView(as: Int32.self).withUnsafeMutablePointer { pointer, _, _ in
                    for index in values.indices { pointer[index] = values[index] }
                }
                return array
            default:
                throw VoxCPMGenerationError.invalidInput(label)
            }
        }

        private static func floats(_ array: NDArray, _ tensor: VoxCPMSignature.Tensor, _ function: VoxCPMFunction)
            throws -> [Float]
        {
            let shape = array.shape
            let count = shape.reduce(1, *)
            guard shape == tensor.shape, array.interleaveLayout == nil else {
                throw VoxCPMGenerationError.invalidOutput(function.rawValue + "." + tensor.name)
            }
            func offset(_ index: Int, _ strides: Span<Int>) -> Int {
                var rest = index
                var offset = 0
                for axis in shape.indices.reversed() {
                    offset += rest % shape[axis] * strides[axis]
                    rest /= shape[axis]
                }
                return offset
            }
            switch array.scalarType {
            case .float32:
                return array.view(as: Float.self).withUnsafePointer { pointer, _, strides in
                    (0..<count).map { pointer[offset($0, strides)] }
                }
            case .float16:
                return array.view(as: Float16.self).withUnsafePointer { pointer, _, strides in
                    (0..<count).map { Float(pointer[offset($0, strides)]) }
                }
            default:
                throw VoxCPMGenerationError.invalidOutput(function.rawValue + "." + tensor.name)
            }
        }

        private static func missing(_ asset: String, _ name: String) -> SpeechError {
            .invalidAsset("The VoxCPM2 \(asset) asset does not provide \(name).")
        }
    }
#endif
