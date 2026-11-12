import ChronosCore
import CoreAI
import Foundation
import Synchronization

final class ChronosNativePredictor: ChronosPredictor {
    let identity: ChronosBundleIdentity
    private struct Resources {
        var snapshot: ChronosBundleSnapshot?
        var model: AIModel?
        var function: InferenceFunction?
    }

    private let resources: Mutex<Resources>

    private init(
        identity: ChronosBundleIdentity, snapshot: ChronosBundleSnapshot, model: AIModel,
        function: InferenceFunction
    ) {
        self.identity = identity
        resources = Mutex(Resources(snapshot: snapshot, model: model, function: function))
    }

    deinit {
        let snapshot = resources.withLock { $0.snapshot }
        resources.withLock {
            $0.function = nil
            $0.model = nil
            $0.snapshot = nil
        }
        withExtendedLifetime(snapshot) {}
    }

    @concurrent
    static func load(
        bundleURL: URL, expectedManifestSHA256: String? = nil,
        computeUnits: ChronosComputeUnits = .gpuPreferred
    ) async throws -> Self {
        try Task.checkCancellation()
        let snapshot = try await ChronosBundleSnapshot.open(bundleURL)
        let bundle = snapshot.bundle
        try validateIdentity(bundle.identity, expectedManifestSHA256: expectedManifestSHA256)
        try Task.checkCancellation()
        let options = specializationOptions(for: computeUnits)
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

    static func specializationOptions(for computeUnits: ChronosComputeUnits) -> SpecializationOptions {
        var options: SpecializationOptions
        switch computeUnits {
        case .gpuPreferred: options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .cpuOnly: options = .cpuOnly
        }
        options.expectFrequentReshapes = false
        return options
    }

    static func validateIdentity(_ identity: ChronosBundleIdentity, expectedManifestSHA256: String?) throws {
        guard expectedManifestSHA256 == nil || identity.manifest.sha256 == expectedManifestSHA256 else {
            throw ChronosError.invalidBundle("managed manifest identity")
        }
    }

    @concurrent
    func predict(_ context: ChronosContext) async throws -> ChronosPrediction {
        defer { withExtendedLifetime(self) {} }
        try Task.checkCancellation()
        let function = try resources.withLock {
            guard let function = $0.function else { throw ChronosError.invalidModel }
            return function
        }
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
