import Foundation
import ImageGenerationCore
import ModelLibraryFeature

public protocol ImageTraceSink: Actor {
    func record(name: String, values: [String: ImageTensorData], contracts: [ImageTensorContract]) async throws
}

actor ImageRuntimePipeline {
    typealias Progress = @MainActor @Sendable (ImageExecutionProgress) -> Void
    typealias Permission = @MainActor @Sendable () -> Bool
    typealias BundleLoader = @Sendable (ValidatedModelResource, ImageMemoryBudget) throws -> ImageRuntimeBundle
    typealias PromptPreparer = @Sendable (URL, String, Bool) throws -> ImagePromptLayout

    let resource: ValidatedModelResource
    let budget: ImageMemoryBudget
    let backend: any ImageComponentBackend
    let permission: Permission
    let memory: @Sendable () -> ImageMemoryObservation
    let loadBundle: BundleLoader
    let preparePrompt: PromptPreparer
    private var active: UUID?
    private var admission: Permission?
    private var stopping = false
    private var released = false
    private var component: ImageModelComponent?
    private var releaseFailure: String?
    private var waiters: [CheckedContinuation<Void, any Error>] = []
    private(set) var report: ImageExecutionReport?

    init(
        resource: ValidatedModelResource, budget: ImageMemoryBudget, backend: any ImageComponentBackend,
        permission: @escaping Permission, memory: @escaping @Sendable () -> ImageMemoryObservation,
        loadBundle: @escaping BundleLoader = ImageRuntimeBundle.load,
        preparePrompt: @escaping PromptPreparer = { directory, prompt, reference in
            try ImageGenerationTokenizer(directory: directory).prepare(prompt, hasReference: reference)
        }
    ) {
        self.resource = resource
        self.budget = budget
        self.backend = backend
        self.permission = permission
        self.memory = memory
        self.loadBundle = loadBundle
        self.preparePrompt = preparePrompt
    }

    func generate(
        _ request: ImageGenerationRequest, trace: (any ImageTraceSink)? = nil,
        isCurrent: @escaping Permission = { true }, progress: Progress
    ) async throws -> ImageGenerationResult {
        guard !released else { throw ImageExecutionError.released }
        guard active == nil, !stopping else { throw ImageExecutionError.busy }
        if let releaseFailure { throw ImageExecutionError.releaseUnconfirmed(releaseFailure) }
        active = request.id
        admission = isCurrent
        report = ImageExecutionReport(
            requestID: request.id, profileID: budget.profileID, manifestSHA256: budget.manifestSHA256,
            noiseSource: request.initialNoise == nil ? ImageGenerationNoise.version : "supplied-float32-packed",
            diagnosticsEnabled: trace != nil)
        let start = ContinuousClock.now
        let outcome: Result<ImageGenerationPixels, any Error>
        do {
            try request.validate()
            try await checkpoint()
            try await emit(.preparing, progress: progress)
            outcome = .success(try await perform(request, trace: trace, progress: progress))
        } catch {
            outcome = .failure(error)
        }
        var failure: (any Error)?
        do {
            try await releaseComponent()
        } catch {
            releaseFailure = String(describing: error)
            failure = ImageExecutionError.releaseUnconfirmed(String(describing: error))
        }
        if failure == nil {
            do { try await checkpoint() } catch { failure = error }
        }
        if case .failure(let original) = outcome {
            if let cleanup = failure as? ImageExecutionError, case .releaseUnconfirmed = cleanup {
                failure = ImageExecutionError.releaseUnconfirmed("Primary: \(original); cleanup: \(cleanup)")
            } else {
                failure = original
            }
        }
        report?.totalSeconds = elapsed(since: start)
        report?.status =
            releaseFailure != nil
            ? .releaseUnconfirmed
            : failure == nil ? .completed : failure is CancellationError ? .cancelled : .failed
        report?.error = failure.map { String(describing: $0) }
        active = nil
        admission = nil
        stopping = false
        finishWaiters()
        if let failure { throw failure }
        guard case .success(let pixels) = outcome, let report else { throw ImageExecutionError.invalidResource }
        return ImageGenerationResult(requestID: request.id, pixels: pixels, report: report)
    }

    func stop(releasing: Bool = false) async throws {
        if releasing { released = true }
        if active != nil || stopping {
            stopping = true
            return try await withCheckedThrowingContinuation { waiters.append($0) }
        }
        stopping = true
        do {
            try await releaseComponent()
            releaseFailure = nil
        } catch {
            releaseFailure = String(describing: error)
        }
        stopping = false
        finishWaiters()
        if let releaseFailure { throw ImageExecutionError.releaseUnconfirmed(releaseFailure) }
    }

    private func finishWaiters() {
        let pending = waiters
        waiters.removeAll()
        for waiter in pending {
            if let releaseFailure {
                waiter.resume(throwing: ImageExecutionError.releaseUnconfirmed(releaseFailure))
            } else {
                waiter.resume()
            }
        }
    }

    private func checkpoint() async throws {
        try Task.checkCancellation()
        guard !stopping, !released else { throw CancellationError() }
        guard let admission, await admission() else { throw CancellationError() }
        try Task.checkCancellation()
        guard !stopping, !released else { throw CancellationError() }
        let permitted = await permission()
        try Task.checkCancellation()
        guard !stopping, !released else { throw CancellationError() }
        guard permitted else { throw ImageExecutionError.permissionDenied }
    }

    private func emit(_ phase: ImageExecutionPhase, progress: Progress) async throws {
        try await checkpoint()
        guard let active else { throw ImageExecutionError.released }
        await progress(ImageExecutionProgress(requestID: active, phase: phase))
        try await checkpoint()
    }

    private func perform(
        _ request: ImageGenerationRequest, trace: (any ImageTraceSink)?, progress: Progress
    ) async throws -> ImageGenerationPixels {
        let bundle = try loadBundle(resource, budget)
        try await checkpoint()
        let layout = try preparePrompt(bundle.tokenizerDirectory, request.prompt, request.reference != nil)
        let structure = try ImageDenoiserLayout(
            imageSlots: layout.imageSlots, capacity: bundle.profile.workload.jointCapacity)
        let text = try await encodeText(
            bundle, layout: layout, image: request.reference, trace: trace, progress: progress)
        let reference = try await encodeReference(bundle, image: request.reference, trace: trace, progress: progress)
        let latents = try await denoise(
            bundle, request: request, structure: structure, text: text, reference: reference, trace: trace,
            progress: progress)
        let precision = try descriptor(.vaeDecoder, bundle: bundle).computeDtype
        let decoded = try bundle.scaling.denormalize(latents, precision: precision)
        try await open(.vaeDecoder, bundle: bundle, progress: progress)
        let output = try await call(
            .vaeDecoder, bundle: bundle, values: ["value": .floats(decoded)], trace: trace, progress: progress)
        try await close(.vaeDecoder, progress: progress)
        return try ImageGenerationPixels.decodeVAE(output.floating("result"))
    }

    private func encodeText(
        _ bundle: ImageRuntimeBundle, layout: ImagePromptLayout, image: ImageGenerationPixels?,
        trace: (any ImageTraceSink)?, progress: Progress
    ) async throws -> [Float] {
        var vision: [[Float]]?
        if let image {
            let values: [String: ImageTensorData] = ["pixel_values": .floats(try image.visionInput())]
            try await open(.vision, bundle: bundle, progress: progress)
            let outputs = try await call(.vision, bundle: bundle, values: values, trace: trace, progress: progress)
            try await close(.vision, progress: progress)
            vision = try ["merged", "deepstack_0", "deepstack_1", "deepstack_2"].map(outputs.floating)
        }
        let conditioning = try ImageTextConditioning(layout: layout, visionOutputs: vision)
        let values: [String: ImageTensorData] = [
            "input_ids": .integers(layout.paddedTokens), "vision": .floats(conditioning.vision),
            "deepstack": .floats(conditioning.deepstack), "cosine": .floats(layout.textCosine),
            "sine": .floats(layout.textSine), "mask": .mask(layout.textMask), "visual_mask": .mask(layout.visualMask),
        ]
        try await open(.text, bundle: bundle, progress: progress)
        let outputs = try await call(.text, bundle: bundle, values: values, trace: trace, progress: progress)
        try await close(.text, progress: progress)
        return try layout.condition(fromHidden: outputs.floating("hidden"))
    }

    private func encodeReference(
        _ bundle: ImageRuntimeBundle, image: ImageGenerationPixels?, trace: (any ImageTraceSink)?,
        progress: Progress
    ) async throws -> [Float]? {
        guard let image else { return nil }
        let values: [String: ImageTensorData] = ["value": .floats(try image.vaeInput())]
        try await open(.vaeEncoder, bundle: bundle, progress: progress)
        let output = try await call(.vaeEncoder, bundle: bundle, values: values, trace: trace, progress: progress)
        try await close(.vaeEncoder, progress: progress)
        let precision = try descriptor(.vaeEncoder, bundle: bundle).computeDtype
        let denoiserPrecision = try descriptor(.denoiser, bundle: bundle).computeDtype
        return try bundle.scaling.normalize(output.floating("result"), precision: precision).map(
            denoiserPrecision.rounded)
    }

    private func denoise(
        _ bundle: ImageRuntimeBundle, request: ImageGenerationRequest, structure: ImageDenoiserLayout,
        text: [Float], reference: [Float]?, trace: (any ImageTraceSink)?, progress: Progress
    ) async throws -> [Float] {
        let precision = try descriptor(.denoiser, bundle: bundle).computeDtype
        let noise = try request.initialNoise ?? ImageGenerationNoise.make(seed: request.seed)
        var latents = try ImageEuler.initialLatents(noise: noise, precision: precision)
        try await save("initial-latents", floats: latents, shape: [1, 1_024, 64], trace: trace)
        try await open(.denoiser, bundle: bundle, progress: progress)
        for index in 0..<8 {
            let values: [String: ImageTensorData] = [
                "latents": .floats(try structure.packedLatents(target: latents, reference: reference)),
                "text": .floats(text),
                "timestep": .floats([try ImageEuler.modelTimestep(index: index, precision: precision)]),
                "text_index": .integers(structure.textIndex), "image_index": .integers(structure.imageIndex),
                "image_mask": .mask(structure.imageMask), "cosine": .floats(structure.cosine),
                "sine": .floats(structure.sine),
                "mask": .mask(structure.attentionMask), "target": .mask(structure.targetMask),
            ]
            let outputs = try await call(
                .denoiser, bundle: bundle, values: values, step: index, trace: trace, progress: progress)
            let prediction = try structure.prediction(fromOutput: outputs.floating("noise"))
            latents = try ImageEuler.step(latents: latents, prediction: prediction, index: index, precision: precision)
            try await save("latent-\(index)", floats: latents, shape: [1, 1_024, 64], trace: trace)
        }
        try await close(.denoiser, progress: progress)
        return latents
    }

    private func open(_ kind: ImageModelComponent, bundle: ImageRuntimeBundle, progress: Progress) async throws {
        guard component == nil else { throw ImageExecutionError.busy }
        try await emit(.loading(kind), progress: progress)
        let before = memory()
        let reserve = try reserve(kind, phase: .loading, observation: before)
        let description = try descriptor(kind, bundle: bundle)
        component = kind
        let start = ContinuousClock.now
        do {
            let load = try await withTaskCancellationShield {
                try await backend.load(directory: bundle.directory, component: description)
            }
            report?.phases.append(
                ImagePhaseMetric(
                    component: kind, operation: .loading, seconds: elapsed(since: start), before: before,
                    after: memory(),
                    reserveBytes: reserve, additionalBudgetBytes: budget.components[kind]?.loadingAdditionalBytes,
                    load: load))
        } catch {
            report?.phases.append(
                ImagePhaseMetric(
                    component: kind, operation: .loading, seconds: elapsed(since: start), before: before,
                    after: memory(),
                    reserveBytes: reserve, additionalBudgetBytes: budget.components[kind]?.loadingAdditionalBytes,
                    error: String(describing: error)))
            throw error
        }
        try await checkpoint()
    }

    private func call(
        _ kind: ImageModelComponent, bundle: ImageRuntimeBundle, values: [String: ImageTensorData], step: Int? = nil,
        trace: (any ImageTraceSink)?, progress: Progress
    ) async throws -> [String: ImageTensorData] {
        guard component == kind else { throw ImageExecutionError.invalidResource }
        let description = try descriptor(kind, bundle: bundle)
        try validate(values, contracts: description.inputs)
        let label = kind.rawValue + (step.map { "-\($0)" } ?? "")
        try await trace?.record(name: label + "-inputs", values: values, contracts: description.inputs)
        try await emit(.running(kind, step: step), progress: progress)
        let before = memory()
        let reserve = try reserve(kind, phase: .inference, observation: before, step: step)
        let start = ContinuousClock.now
        let outputs: [String: ImageTensorData]
        do {
            outputs = try await withTaskCancellationShield { try await backend.call(values) }
            report?.phases.append(
                ImagePhaseMetric(
                    component: kind, operation: .inference, step: step, seconds: elapsed(since: start), before: before,
                    after: memory(),
                    reserveBytes: reserve, additionalBudgetBytes: budget.components[kind]?.inferenceAdditionalBytes))
        } catch {
            report?.phases.append(
                ImagePhaseMetric(
                    component: kind, operation: .inference, step: step, seconds: elapsed(since: start), before: before,
                    after: memory(),
                    reserveBytes: reserve, additionalBudgetBytes: budget.components[kind]?.inferenceAdditionalBytes,
                    error: String(describing: error)))
            throw error
        }
        try await trace?.record(name: label + "-outputs", values: outputs, contracts: description.outputs)
        try validate(outputs, contracts: description.outputs)
        try await checkpoint()
        return outputs
    }

    private func close(_ kind: ImageModelComponent, progress: Progress) async throws {
        try await emit(.releasing(kind), progress: progress)
        try await releaseComponent()
        try await checkpoint()
    }

    private func releaseComponent() async throws {
        guard let kind = component else { return }
        let start = ContinuousClock.now
        let before = memory()
        do {
            try await withTaskCancellationShield { try await backend.unload() }
            component = nil
            report?.phases.append(
                ImagePhaseMetric(
                    component: kind, operation: .release, seconds: elapsed(since: start), before: before,
                    after: memory()))
        } catch {
            report?.phases.append(
                ImagePhaseMetric(
                    component: kind, operation: .release, seconds: elapsed(since: start), before: before,
                    after: memory(),
                    error: String(describing: error)))
            throw error
        }
    }

    private func reserve(
        _ kind: ImageModelComponent, phase: ImageMemoryPhase, observation: ImageMemoryObservation, step: Int? = nil
    ) throws -> UInt64 {
        do {
            return try budget.check(kind, phase: phase, observation: observation)
        } catch {
            report?.phases.append(
                ImagePhaseMetric(
                    component: kind, operation: phase == .loading ? .loading : .inference, step: step, seconds: 0,
                    before: observation, after: observation,
                    additionalBudgetBytes: phase == .loading
                        ? budget.components[kind]?.loadingAdditionalBytes
                        : budget.components[kind]?.inferenceAdditionalBytes,
                    error: String(describing: error)))
            throw error
        }
    }

    private func descriptor(_ kind: ImageModelComponent, bundle: ImageRuntimeBundle) throws
        -> ImageGenerationProfile.Component
    {
        guard let result = bundle.profile.components[kind.rawValue] else { throw ImageExecutionError.invalidResource }
        return result
    }

    private func validate(_ values: [String: ImageTensorData], contracts: [ImageTensorContract]) throws {
        guard Set(values.keys) == Set(contracts.map(\.name)) else { throw ImageExecutionError.invalidResource }
        for contract in contracts {
            guard let value = values[contract.name] else { throw ImageExecutionError.invalidTensor(contract.name) }
            try value.validate(contract)
        }
    }

    private func save(_ name: String, floats: [Float], shape: [Int], trace: (any ImageTraceSink)?) async throws {
        guard let trace else { return }
        let description = ImageTensorContract(name: "value", shape: shape, dtype: .float32, layout: "row-major")
        try await trace.record(name: name, values: ["value": .floats(floats)], contracts: [description])
        try await checkpoint()
    }

    private func elapsed(since start: ContinuousClock.Instant) -> Double {
        (ContinuousClock.now - start) / .seconds(1)
    }
}

extension Dictionary where Key == String, Value == ImageTensorData {
    fileprivate func floating(_ name: String) throws -> [Float] {
        guard let value = self[name] else { throw ImageExecutionError.invalidTensor(name) }
        return try value.floats()
    }
}
