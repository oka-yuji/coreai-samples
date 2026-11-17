import Foundation
import SpeechCore
import Testing
import VoxCPMCore

@testable import SpeechAssets
@testable import SpeechRuntime

@Suite(.serialized)
struct VoxCPMNativeTests {
    struct SetupError: Error, Sendable {
        let message: String
    }

    static let environment = ProcessInfo.processInfo.environment
    static let enabled = ["COREAI_VOXCPM2_BUNDLES", "COREAI_VOXCPM2_REFERENCE", "COREAI_VOXCPM2_OUTPUT"]
        .allSatisfy { environment[$0] != nil }
    static let criterion = 0.001
    static let output = Result(catching: makeOutput)

    static func makeOutput() throws(SetupError) -> URL {
        guard let path = environment["COREAI_VOXCPM2_OUTPUT"] else { throw SetupError(message: "No output folder") }
        let url = URL(filePath: path, directoryHint: .isDirectory)
        do {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        } catch {
            throw SetupError(message: "The output folder must be new: \(error)")
        }
        return url
    }

    static func folder(_ name: String) throws -> URL {
        let url = try output.get().appending(path: name, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    static func bundle(_ name: String) throws -> URL {
        URL(filePath: try #require(environment["COREAI_VOXCPM2_BUNDLES"]), directoryHint: .isDirectory)
            .appending(path: name, directoryHint: .isDirectory)
    }

    static func references() throws -> (index: VoxCPMReference.Index, root: URL) {
        let root = URL(filePath: try #require(environment["COREAI_VOXCPM2_REFERENCE"]), directoryHint: .isDirectory)
        return (try VoxCPMReference.index(root), root)
    }

    static func write(_ object: [String: Any], _ url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
            .write(to: url, options: .withoutOverwriting)
    }

    static func write(_ values: [Float], _ url: URL) throws {
        var data = Data(capacity: values.count * 4)
        for value in values { withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) } }
        try data.write(to: url, options: .withoutOverwriting)
    }

    static func statistics(_ value: VoxCPMLoadStatistics) -> [String: Any] {
        ["openSeconds": value.openSeconds, "assetSeconds": value.assetSeconds, "totalSeconds": value.totalSeconds]
    }

    static func number(_ value: Double?) -> Any {
        value.map { $0 as Any } ?? "nonfinite"
    }

    @Test(
        "VPL-06 Each recorded host call gives the same outputs from the Swift parts within 0.001", .enabled(if: enabled)
    )
    func perCall() async throws {
        let folder = try Self.folder("per-call")
        let (index, root) = try Self.references()
        let loaded = try await VoxCPMPipeline.load(bundle: Self.bundle("fp32"))
        let assets = loaded.pipeline.components
        var sentences: [String: Any] = [:]
        var passed = true
        for name in index.sentences {
            let reference = try VoxCPMReference(root.appending(path: name))
            var states = try assets.makeStates()
            var maximum: [String: Double] = [:]
            var counts: [String: Int] = [:]
            var failures: [[String: Any]] = []
            for call in reference.record.calls {
                let function = try #require(VoxCPMFunction(rawValue: call.name))
                let signature = try #require(assets.signatures[function])
                var inputs: [String: VoxCPMInput] = [:]
                for tensor in signature.inputs {
                    inputs[tensor.name] = try reference.input("\(call.name).in.\(tensor.name)", call.index)
                }
                let outputs = try await assets.call(function, inputs, states: &states)
                for tensor in signature.outputs {
                    let key = "\(call.name).\(tensor.name)"
                    let error = VoxCPMReference.error(
                        try reference.floats("\(call.name).out.\(tensor.name)", call.index), outputs[tensor.name] ?? [])
                    counts[key, default: 0] += 1
                    maximum[key] = max(maximum[key] ?? 0, error ?? .greatestFiniteMagnitude)
                    if !((error ?? .infinity) <= Self.criterion) {
                        failures.append([
                            "call": call.name, "index": call.index, "output": tensor.name, "error": Self.number(error),
                        ])
                    }
                }
            }
            passed = passed && failures.isEmpty
            sentences[name] = [
                "calls": reference.record.calls.count, "comparisons": counts.values.reduce(0, +), "counts": counts,
                "maximum": maximum, "largest": maximum.values.max() ?? 0, "failures": failures,
            ]
            #expect(failures.isEmpty, "\(name)")
        }
        try Self.write(
            [
                "passed": passed, "criterion": Self.criterion, "load": Self.statistics(loaded.statistics),
                "sentences": sentences,
            ], folder.appending(path: "per-call.json"))
    }

    @Test(
        "VPL-06 With the recorded latent at each patch the solver latent, hidden state, stop logits, audio and stop match",
        .enabled(if: enabled))
    func latentDriven() async throws {
        let folder = try Self.folder("latent-driven")
        let (index, root) = try Self.references()
        let loaded = try await VoxCPMPipeline.load(bundle: Self.bundle("fp32"))
        var sentences: [String: Any] = [:]
        var passed = true
        for name in index.sentences {
            let reference = try VoxCPMReference(root.appending(path: name))
            let latents = try (0..<reference.record.patches).map { try reference.floats("patch.latent", $0) }
            let recorder = VoxCPMPatchRecorder(replacements: latents)
            let speech = try await loaded.pipeline.generate(
                ids: reference.record.ids, seed: reference.record.seed, observer: recorder.observe)
            let size = index.samplesPerPatch
            var maximum: [String: Double] = [:]
            var failures: [[String: Any]] = []
            var noiseEqual = true
            for record in recorder.patches where record.index < reference.record.patches {
                let patch = record.index
                let audio =
                    patch < speech.patchCount
                    ? Array(speech.samples[(patch * size)..<((patch + 1) * size)]) : []
                let errors: [(String, Double?)] = [
                    (
                        "solver_latent",
                        VoxCPMReference.error(try reference.floats("patch.latent", patch), record.latent)
                    ),
                    (
                        "dit_hidden",
                        VoxCPMReference.error(try reference.floats("patch.dit_hidden", patch), record.ditHidden)
                    ),
                    (
                        "stop_logits",
                        VoxCPMReference.error(try reference.floats("patch.stop_logits", patch), record.stopLogits)
                    ),
                    ("cond", VoxCPMReference.error(try reference.floats("patch.cond", patch), record.condition)),
                    ("audio", VoxCPMReference.error(try reference.floats("vae.main.out.audio", patch), audio)),
                ]
                let equal = try reference.slice("patch.noise", patch) == record.noise.map(\.bitPattern)
                noiseEqual = noiseEqual && equal
                for (key, error) in errors {
                    maximum[key] = max(maximum[key] ?? 0, error ?? .greatestFiniteMagnitude)
                    if !((error ?? .infinity) <= Self.criterion) {
                        failures.append(["patch": patch, "value": key, "error": Self.number(error)])
                    }
                }
            }
            let stopMatches =
                speech.patchCount == reference.record.patches && speech.stoppedPatch == reference.record.stoppedPatch
            let joined = VoxCPMReference.error(try reference.floats("audio"), speech.samples)
            let sentencePassed =
                failures.isEmpty && stopMatches && noiseEqual && (joined ?? .infinity) <= Self.criterion
            passed = passed && sentencePassed
            let arrays = folder.appending(path: name, directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: arrays, withIntermediateDirectories: false)
            try Self.write(recorder.patches.flatMap(\.latent), arrays.appending(path: "solver_latent.f32"))
            sentences[name] = [
                "passed": sentencePassed, "patches": speech.patchCount, "stoppedPatch": speech.stoppedPatch,
                "referencePatches": reference.record.patches, "referenceStoppedPatch": reference.record.stoppedPatch,
                "noiseBitEqual": noiseEqual, "maximum": maximum, "joinedAudio": Self.number(joined),
                "failures": failures,
            ]
            #expect(sentencePassed, "\(name)")
        }
        try Self.write(
            [
                "passed": passed, "criterion": Self.criterion, "load": Self.statistics(loaded.statistics),
                "sentences": sentences,
            ], folder.appending(path: "latent-driven.json"))
    }

    @Test(
        "The FP32 free run from seed 1708 is recorded against the host record and stops with its noise and finite values",
        .enabled(if: enabled))
    func freeRunning() async throws {
        let folder = try Self.folder("free-running")
        let (index, root) = try Self.references()
        let loaded = try await VoxCPMPipeline.load(bundle: Self.bundle("fp32"))
        var sentences: [String: Any] = [:]
        var passed = true
        for name in index.sentences {
            let reference = try VoxCPMReference(root.appending(path: name))
            let recorder = VoxCPMPatchRecorder()
            let clock = ContinuousClock()
            let start = clock.now
            var speech: VoxCPMSpeech?
            var failure: String?
            do {
                speech = try await loaded.pipeline.generate(
                    ids: reference.record.ids, seed: reference.record.seed, observer: recorder.observe)
            } catch {
                failure = "\(error)"
            }
            let seconds = (clock.now - start) / .seconds(1)
            let patches = recorder.patches
            var rows: [[String: Any]] = []
            var noiseEqual = true
            var firstOutside: Int?
            var largest = 0.0
            for record in patches where record.index < reference.record.patches {
                let patch = record.index
                let latent = VoxCPMReference.error(try reference.floats("patch.latent", patch), record.latent)
                let equal = try reference.slice("patch.noise", patch) == record.noise.map(\.bitPattern)
                noiseEqual = noiseEqual && equal
                largest = max(largest, latent ?? .greatestFiniteMagnitude)
                if firstOutside == nil, !((latent ?? .infinity) <= Self.criterion) { firstOutside = patch }
                rows.append([
                    "patch": patch, "latent": Self.number(latent), "noiseBitEqual": equal,
                    "ditHidden": Self.number(
                        VoxCPMReference.error(try reference.floats("patch.dit_hidden", patch), record.ditHidden)),
                    "cond": Self.number(
                        VoxCPMReference.error(try reference.floats("patch.cond", patch), record.condition)),
                    "stopLogits": Self.number(
                        VoxCPMReference.error(try reference.floats("patch.stop_logits", patch), record.stopLogits)),
                ])
            }
            let arrays = folder.appending(path: name, directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: arrays, withIntermediateDirectories: false)
            try Self.write(patches.flatMap(\.noise), arrays.appending(path: "noise.f32"))
            try Self.write(patches.flatMap(\.ditHidden), arrays.appending(path: "dit_hidden.f32"))
            try Self.write(patches.flatMap(\.condition), arrays.appending(path: "cond.f32"))
            try Self.write(patches.flatMap(\.latent), arrays.appending(path: "latent.f32"))
            try Self.write(patches.flatMap(\.stopLogits), arrays.appending(path: "stop_logits.f32"))
            let samples = speech?.samples ?? []
            let finite =
                (samples + patches.flatMap { $0.noise + $0.ditHidden + $0.condition + $0.latent + $0.stopLogits })
                .allSatisfy(\.isFinite)
            let stopMatches = speech.map {
                $0.patchCount == reference.record.patches && $0.stoppedPatch == reference.record.stoppedPatch
            }
            let sentencePassed = failure == nil && speech != nil && noiseEqual && finite
            passed = passed && sentencePassed
            sentences[name] = [
                "passed": sentencePassed, "finish": failure ?? "stop", "patches": patches.count,
                "stoppedPatch": speech.map { $0.stoppedPatch as Any } ?? NSNull(),
                "referencePatches": reference.record.patches, "referenceStoppedPatch": reference.record.stoppedPatch,
                "stopPatchMatches": stopMatches.map { $0 as Any } ?? NSNull(), "audioSamples": samples.count,
                "audioSeconds": Double(samples.count) / Double(loaded.pipeline.contract.sampleRate),
                "noiseBitEqual": noiseEqual, "noiseComparedPatches": rows.count, "finite": finite,
                "firstPatchOutside": firstOutside.map { $0 as Any } ?? NSNull(), "largestLatentError": largest,
                "seconds": seconds, "rows": rows,
            ]
            #expect(sentencePassed, "\(name)")
        }
        try Self.write(
            [
                "passed": passed, "criterion": Self.criterion, "load": Self.statistics(loaded.statistics),
                "verdict": "The noise, the finite values and the stop decide the verdict.",
                "recordedOnly":
                    "The latent differences, the first patch outside the criterion, the largest difference and the stop patch are recorded without a verdict. A free run is not evidence that the Swift pipeline follows the host record or the original trajectory.",
                "sentences": sentences,
            ], folder.appending(path: "free-running.json"))
    }

    @Test(
        "VPL-07 Each configuration generates audio whose length matches the stop patch without non-finite samples",
        .enabled(if: enabled))
    func profiles() async throws {
        let folder = try Self.folder("profiles")
        let (index, root) = try Self.references()
        let references = try index.sentences.map { try VoxCPMReference(root.appending(path: $0)) }
        var configurations: [String: Any] = [:]
        var passed = true
        for profile in VoxCPMBundleProfile.allCases {
            let loaded = try await VoxCPMPipeline.load(bundle: Self.bundle(profile.rawValue))
            #expect(loaded.pipeline.contract.profile == profile)
            let clock = ContinuousClock()
            let warmStart = clock.now
            let warm = try await loaded.pipeline.generate(
                ids: references[0].record.ids, seed: references[0].record.seed)
            let warmSeconds = (clock.now - warmStart) / .seconds(1)
            var rows: [String: Any] = [:]
            for reference in references {
                let start = clock.now
                let speech = try await loaded.pipeline.generate(ids: reference.record.ids, seed: reference.record.seed)
                let seconds = (clock.now - start) / .seconds(1)
                let audioSeconds = Double(speech.samples.count) / Double(speech.sampleRate)
                let lengthMatches =
                    speech.samples.count == speech.patchCount * loaded.pipeline.contract.samplesPerPatch
                    && speech.stoppedPatch == speech.patchCount - 1
                let finite = speech.samples.allSatisfy(\.isFinite)
                passed = passed && lengthMatches && finite
                #expect(lengthMatches && finite, "\(profile.rawValue) \(reference.record.name)")
                let peak = speech.samples.map { abs($0) }.max() ?? 0
                rows[reference.record.name] = [
                    "patches": speech.patchCount, "stoppedPatch": speech.stoppedPatch,
                    "referenceFP32StoppedPatch": reference.record.stoppedPatch, "audioSeconds": audioSeconds,
                    "seconds": seconds, "millisecondsPerPatch": seconds / Double(speech.patchCount) * 1000,
                    "secondsPerAudioSecond": seconds / audioSeconds, "lengthMatches": lengthMatches, "finite": finite,
                    "peak": Double(peak),
                ]
            }
            configurations[profile.rawValue] = [
                "load": Self.statistics(loaded.statistics), "warmupSeconds": warmSeconds,
                "warmupPatches": warm.patchCount, "sentences": rows, "memoryAfter": VoxCPMMemory.capture().json,
            ]
        }
        try Self.write(
            [
                "passed": passed, "configurations": configurations, "memoryAtEnd": VoxCPMMemory.capture().json,
                "memoryBoundary":
                    "Point samples of the test process; the lifetime peak covers every test in the process.",
            ], folder.appending(path: "profiles.json"))
    }
}
