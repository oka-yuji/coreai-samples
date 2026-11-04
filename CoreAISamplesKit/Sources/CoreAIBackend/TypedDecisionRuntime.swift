import CoreAI
import Foundation

struct TypedDecisionRuntime: TypedDecisionBackbone {
    let model: AIModel
    let function: InferenceFunction
    let hiddenDescriptor: NDArrayDescriptor
    let stateDescriptors: [NDArrayDescriptor]
    let hiddenSize: Int
    let stateNames: [String]
    let limit: Int

    static func load(bundle: TypedDecisionBundle) async throws -> (runtime: Self, cached: Bool) {
        do {
            let summary = try AIModelAsset(contentsOf: bundle.assetURL).summary(includingStatistics: false)
            guard
                summary?.functions.first(where: { $0.name == bundle.backboneFunction })?.outputs.contains(
                    where: { $0.name == bundle.backboneOutput }) == true
            else { throw TypedDecisionError.invalidBundle("The asset does not return \(bundle.backboneOutput).") }
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as TypedDecisionError {
            throw error
        } catch {
            throw TypedDecisionError.invalidBundle(error.localizedDescription)
        }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let cached = try AIModelCache.default.model(for: bundle.assetURL, options: options)
        let model: AIModel
        if let cached {
            model = cached
        } else {
            model = try await AIModel.specialize(contentsOf: bundle.assetURL, options: options)
        }
        try Task.checkCancellation()
        guard let descriptor = model.functionDescriptor(for: bundle.backboneFunction),
            let function = try model.loadFunction(named: bundle.backboneFunction)
        else { throw TypedDecisionError.invalidBundle("The asset has no \(bundle.backboneFunction) function.") }
        let hiddenSize = bundle.hiddenSize
        guard Set(descriptor.inputNames) == Set(bundle.backboneInputs),
            bundle.backboneInputs.allSatisfy({
                matches(descriptor.inputDescriptor(of: $0), type: .int32, shape: [1, -1])
            }),
            matches(
                descriptor.outputDescriptor(of: bundle.backboneOutput), type: .float16, shape: [1, -1, hiddenSize]),
            case .ndArray(let hidden) = descriptor.outputDescriptor(of: bundle.backboneOutput),
            descriptor.stateNames.count == 4, Set(descriptor.stateNames) == Set(bundle.stateNames)
        else { throw TypedDecisionError.invalidBundle("The asset descriptors do not match decision.json.") }
        var states: [NDArrayDescriptor] = []
        for name in bundle.stateNames {
            guard case .ndArray(let state) = descriptor.stateDescriptor(of: name), state.interleaveLayout == nil,
                state.shape.filter({ $0 < 0 }).count <= 1
            else { throw TypedDecisionError.invalidBundle("Unsupported state layout: \(name).") }
            states.append(state)
        }
        let runtime = Self(
            model: model, function: function, hiddenDescriptor: hidden, stateDescriptors: states,
            hiddenSize: hiddenSize, stateNames: bundle.stateNames, limit: bundle.limits.inputTokens)
        return (runtime, cached != nil)
    }

    func forward(_ ids: [Int], capacity: Int) async throws -> [Float] {
        guard !ids.isEmpty, ids.allSatisfy({ (0...Int(Int32.max)).contains($0) }) else {
            throw TypedDecisionError.invalidOptions("The backbone takes token IDs from 0 to \(Int32.max).")
        }
        guard ids.count <= capacity, capacity <= limit else {
            throw TypedDecisionError.tooManyTokens(count: ids.count, limit: min(capacity, limit))
        }
        let count = ids.count
        let inputs = [
            "input_ids": Self.integers(ids.map(Int32.init), count: count),
            "position_ids": Self.integers((0..<count).map(Int32.init), count: count),
        ]
        var hidden = NDArray(descriptor: hiddenDescriptor.resolvingDynamicDimensions([1, count, hiddenSize]))
        var first = state(0, capacity: capacity)
        var second = state(1, capacity: capacity)
        var third = state(2, capacity: capacity)
        var fourth = state(3, capacity: capacity)
        let function = function
        try Task.checkCancellation()
        var stateViews = InferenceFunction.MutableViews()
        stateViews.insert(&first, for: stateNames[0])
        stateViews.insert(&second, for: stateNames[1])
        stateViews.insert(&third, for: stateNames[2])
        stateViews.insert(&fourth, for: stateNames[3])
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&hidden, for: "hidden_states")
        _ = try await function.run(inputs: inputs, states: stateViews, outputViews: outputViews)
        try Task.checkCancellation()
        var output = [Float](repeating: 0, count: hiddenSize)
        hidden.view(as: Float16.self).withUnsafePointer { pointer, _, strides in
            let row = (count - 1) * strides[1]
            for column in 0..<hiddenSize {
                output[column] = Float(pointer[row + column * strides[2]])
            }
        }
        return output
    }

    private func state(_ index: Int, capacity: Int) -> NDArray {
        let descriptor = stateDescriptors[index]
        let resolved = descriptor.resolvingDynamicDimensions(descriptor.shape.map { $0 < 0 ? capacity : $0 })
        var array = NDArray(descriptor: resolved)
        let bytes = resolved.minimumByteCount
        array.mutableRawView().withUnsafeMutableBytes { pointer, _, _ in
            _ = pointer.initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
        }
        return array
    }

    private static func matches(_ value: InferenceValue.Descriptor?, type: NDArray.ScalarType, shape: [Int]) -> Bool {
        guard case .ndArray(let array) = value, array.scalarType == type, array.interleaveLayout == nil,
            array.shape.count == shape.count
        else { return false }
        return zip(array.shape, shape).allSatisfy { $1 < 0 ? $0 < 0 : $0 == $1 }
    }

    private static func integers(_ values: [Int32], count: Int) -> NDArray {
        var array = NDArray(shape: [1, count], scalarType: .int32)
        array.mutableView(as: Int32.self).withUnsafeMutablePointer { pointer, _, strides in
            for index in values.indices { pointer[index * strides[1]] = values[index] }
        }
        return array
    }
}
