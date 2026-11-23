import CoreAI
import Foundation

final class AIModelFunctionSource: FunctionSource {
    private let url: URL
    private var function: InferenceFunction?

    init(url: URL) {
        self.url = url
    }

    static func specializationOptions() -> SpecializationOptions {
        var options = SpecializationOptions.default
        options.expectFrequentReshapes = true
        return options
    }

    static func computeUnitName(_ kind: ComputeUnitKind?) -> String {
        guard let kind else { return "default" }
        switch kind {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "neuralEngine"
        @unknown default: return "other"
        }
    }

    static func array(_ tensor: FunctionTensor) throws -> NDArray {
        var array = NDArray(shape: tensor.shape, scalarType: tensor.dtype.scalarType)
        guard array.interleaveLayout == nil,
            FunctionTensor.hasRowMajorLayout(shape: array.shape, strides: array.strides)
        else { throw FunctionError.cannot("write the input \(tensor.name)", "the array is not stored row-major") }
        tensor.data.withUnsafeBytes { source in
            array.mutableRawView().withUnsafeMutableBytes { target, _, _ in
                if let base = source.baseAddress { target.copyMemory(from: base, byteCount: source.count) }
            }
        }
        return array
    }

    static func tensor(_ name: String, _ array: NDArray) throws -> FunctionTensor {
        guard let dtype = FunctionDtype(array.scalarType) else {
            throw FunctionError.cannot("read the output \(name)", "its dtype is not int32, float16 or float32")
        }
        guard array.interleaveLayout == nil,
            FunctionTensor.hasRowMajorLayout(shape: array.shape, strides: array.strides)
        else { throw FunctionError.cannot("read the output \(name)", "the array is not stored row-major") }
        let count = array.shape.reduce(dtype.width, *)
        let data = array.rawView().withUnsafeBytes { pointer, _, _ in Data(bytes: pointer, count: count) }
        return FunctionTensor(name: name, dtype: dtype, shape: array.shape, data: data)
    }

    static func mainHash(in folder: URL) -> String? {
        (try? InputFile.read(folder.appending(path: "main.hash"), label: "main.hash", expectedBytes: nil)).map(hex)
    }

    func describe() throws -> (mainHash: String?, functions: [FunctionSignature]) {
        guard let summary = try AIModelAsset(contentsOf: url).summary(includingStatistics: false) else {
            throw FunctionError.cannot("read the functions of the asset", "no summary could be read from the folder")
        }
        let functions = summary.functions.map { function in
            FunctionSignature(
                name: function.name, inputs: function.inputs.map { .init(name: $0.name, typeName: $0.typeName) },
                stateNames: function.states.map(\.name))
        }
        return (Self.mainHash(in: url), functions)
    }

    func load(_ name: String) async throws -> FunctionLoad {
        let clock = ContinuousClock()
        let start = clock.now
        let options = Self.specializationOptions()
        var queryError: String?
        var cachedModel: AIModel?
        do {
            cachedModel = try AIModelCache.default.model(for: url, options: options)
        } catch {
            queryError = String(describing: error)
        }
        var specializeSeconds = 0.0
        let model: AIModel
        if let cachedModel {
            model = cachedModel
        } else {
            let specializeStart = clock.now
            model = try await AIModel.specialize(contentsOf: url, options: options)
            specializeSeconds = seconds(clock.now - specializeStart)
        }
        guard model.functionNames.contains(name), let loaded = try model.loadFunction(named: name) else {
            throw FunctionError.cannot("load the function \(name)", "the specialized asset has no such function")
        }
        function = loaded
        return FunctionLoad(
            requestedCompute: Self.computeUnitName(options.preferredComputeUnitKind),
            expectFrequentReshapes: options.expectFrequentReshapes, cached: cachedModel != nil,
            cacheQueryError: queryError, specializeS: specializeSeconds, loadS: seconds(clock.now - start))
    }

    func call(_ inputs: [FunctionTensor]) async throws -> [FunctionTensor] {
        guard let function else { throw FunctionError.cannot("run the function", "it is not loaded") }
        var arrays: [String: NDArray] = [:]
        for input in inputs { arrays[input.name] = try Self.array(input) }
        var outputs = try await function.run(inputs: arrays)
        var tensors: [FunctionTensor] = []
        for name in function.descriptor.outputNames {
            guard let array = outputs.remove(name)?.ndArray else {
                throw FunctionError.cannot("read the output \(name)", "the function returned no array for it")
            }
            tensors.append(try Self.tensor(name, array))
        }
        return tensors
    }
}
