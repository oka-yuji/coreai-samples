import Foundation

@testable import SpeechAssets
@testable import SpeechRuntime

actor FakeVoxCPMDriver {
    let started = TestSignal()
    let gate = TestGate()
    let log = LifetimeLog()
    var block = false
    var loadFailure: (any Error)?
    var failure: (any Error)?
    var patches = 3
    var extraSamples = 0
    var factoryCount = 0
    var requests: [[Int32]] = []
    var seeds: [UInt64] = []

    func configure(
        block: Bool = false, loadFailure: (any Error)? = nil, failure: (any Error)? = nil, patches: Int = 3,
        extraSamples: Int = 0
    ) {
        self.block = block
        self.loadFailure = loadFailure
        self.failure = failure
        self.patches = patches
        self.extraSamples = extraSamples
    }

    func makePipeline() throws -> any VoxCPMSpeechPipeline {
        if let loadFailure { throw loadFailure }
        factoryCount += 1
        log.add("factory-\(factoryCount)")
        return FakeVoxCPMSpeechPipeline(lifetime: PipelineLifetime(log: log), driver: self)
    }

    func generate(ids: [Int32], seed: UInt64) async throws -> VoxCPMSpeech {
        requests.append(ids)
        seeds.append(seed)
        started.fire()
        if block {
            block = false
            await gate.wait()
        }
        if let failure { throw failure }
        return VoxCPMSpeech(
            samples: [Float](repeating: 0.25, count: patches * 7680 + extraSamples), sampleRate: 48_000,
            patchCount: patches, stoppedPatch: patches - 1)
    }

    nonisolated func dependencies(
        profile: VoxCPMBundleProfile = .fp32, releasing: @escaping @Sendable () -> Void = {}
    ) -> VoxCPMDependencies {
        VoxCPMDependencies(
            open: { directory in
                self.log.add("open")
                return VoxCPMOpenedBundle(
                    bundle: ValidatedSpeechBundle(
                        directory: directory, assets: [], tokenizer: directory.appending(path: "tokenizer.json")),
                    contract: try VoxCPMPipelineTests.contract(profile))
            },
            tokenizer: { _ in
                self.log.add("tokenizer")
                return FakeVoxCPMTextTokenizer(log: self.log)
            },
            pipeline: { _ in try await self.makePipeline() },
            didRequestRelease: releasing)
    }
}
