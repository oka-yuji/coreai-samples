import ChronosCore
import CoreAI
import Foundation

struct ChronosNativePredictor: ChronosPredictor {
    let identity: ChronosBundleIdentity
    private let snapshot: ChronosBundleSnapshot
    private let model: AIModel
    private let function: InferenceFunction

    @concurrent
    static func load(bundleURL: URL) async throws -> Self {
        try Task.checkCancellation()
        let snapshot = try await ChronosBundleSnapshot.open(bundleURL)
        let bundle = snapshot.bundle
        try Task.checkCancellation()
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = false
        let model = try await AIModel(contentsOf: bundle.url.appendingPathComponent("Model.aimodel"), options: options)
        try Task.checkCancellation()
        guard model.functionNames == ["main"], let descriptor = model.functionDescriptor(for: "main") else {
            throw ChronosError.invalidModel
        }
        try validate(descriptor)
        guard let function = try model.loadFunction(named: "main") else { throw ChronosError.invalidModel }
        try validate(function.descriptor)
        try Task.checkCancellation()
        return Self(identity: bundle.identity, snapshot: snapshot, model: model, function: function)
    }

    @concurrent
    func predict(_ context: ChronosContext) async throws -> ChronosPrediction {
        try Task.checkCancellation()
        try Self.validate(function.descriptor)
        let input = NDArray(scalars: context.paddedValues, shape: [1, context.paddedValues.count])
        var outputs = try await function.run(inputs: ["context": input])
        try Task.checkCancellation()
        guard outputs.count == 1, let array = outputs.remove("quantiles")?.ndArray else {
            throw ChronosError.invalidPrediction
        }
        let result = try Self.extract(array)
        try Task.checkCancellation()
        return result
    }

    static func extract(_ array: borrowing NDArray) throws -> ChronosPrediction {
        try validateOutput(shape: array.shape, scalarType: array.scalarType, interleaveLayout: array.interleaveLayout)
        return try extract(array.view(as: Float.self))
    }

    static func validateOutput(
        shape: [Int], scalarType: NDArray.ScalarType, interleaveLayout: NDArray.InterleaveLayout?
    ) throws {
        guard shape == ChronosContract.outputShape, scalarType == .float32, interleaveLayout == nil else {
            throw ChronosError.invalidPrediction
        }
    }

    static func extract(_ view: borrowing NDArray.View<Float>) throws -> ChronosPrediction {
        guard view.rank == 3, view.shape[0] == 1, view.shape[1] == 21, view.shape[2] == 16,
            view.interleaveLayout == nil
        else { throw ChronosError.invalidPrediction }
        let rows = view.withUnsafePointer { pointer, _, strides in
            (0..<21).map { quantile in
                (0..<16).map { step in pointer[quantile * strides[1] + step * strides[2]] }
            }
        }
        return try ChronosPrediction(quantiles: rows)
    }

    private static func validate(_ descriptor: InferenceFunctionDescriptor) throws {
        guard descriptor.name == "main", descriptor.stateNames.isEmpty,
            descriptor.inputNames == ["context"], descriptor.outputNames == ["quantiles"],
            case .ndArray(let input) = descriptor.inputDescriptor(of: "context"),
            input.shape == [1, -1], input.scalarType == .float32, input.interleaveLayout == nil,
            case .ndArray(let output) = descriptor.outputDescriptor(of: "quantiles"),
            output.shape == ChronosContract.outputShape, output.scalarType == .float32, output.interleaveLayout == nil
        else { throw ChronosError.invalidModel }
    }
}
