#if os(macOS)
    import AVFAudio
    import CryptoKit
    import Darwin
    import Foundation
    import SpeechAssets
    import SpeechCore
    import Testing

    @testable import SpeechRuntime

    @Suite(.serialized)
    struct VoxCPMReferenceWorkflowNativeTests {
        private struct Plan: Decodable {
            struct Sample: Decodable {
                let id: String
                let patches: Int
            }
            let format: Int
            let seed: UInt64
            let text: String
            let referenceWAVManifestSHA256: String
            let pytorchGoldenReportSHA256: String
            let cases: [Sample]
        }

        private struct Voices: Decodable {
            struct File: Decodable {
                let id: String
                let file: String
                let bytes: Int
                let sha256: String
                let pcmSHA256: String
                let rate: Int
                let samples: Int
                enum CodingKeys: String, CodingKey {
                    case id = "case"
                    case file, bytes, sha256, rate, samples
                    case pcmSHA256 = "pcm_sha256"
                }
            }
            let format: Int
            let cases: [File]
        }

        private struct Wave: Encodable {
            let id: String
            let upstreamPatches: Int
            let stopReason: String
            let stoppedPatch: Int
            let patches: Int
            let sampleRate: Int
            let samples: Int
            let audioSeconds: Double
            let elapsedSeconds: Double
            let loadSeconds: Double?
            let modelSerial: Int
            let referenceEncoderCalls: Int
            let wavSHA256: String
        }

        private struct Report: Encodable {
            let format = 1
            let seed = 1708
            let generation = "free"
            var complete = false
            var passed = false
            var manifestSHA256: String?
            var inventorySHA256: String?
            var referenceManifestSHA256: String?
            var upstreamReportSHA256: String?
            var waves = [Wave]()
            var faults = [String]()
            var recovery = [Wave]()
            var released = [Int]()
            var errorType: String?
        }

        private static let prefix = "COREAI_VOXCPM2_REFGEN_"
        private static let enabled = ProcessInfo.processInfo.environment[prefix + "BUNDLE"] != nil
        private static let planSHA = "a3ceb70f768f6a3ecd0829574ec4876d9ad41219964df80946980fe5bf759ba9"
        private static let ids = ["none", "0a67-5", "0a67-10", "1410-5", "1410-10"]

        @Test(
            "VRN-01 The Swift runtime generates five comparison WAVs and recovers after two post-encoder faults",
            .enabled(if: enabled))
        func generateAndRecover() async throws {
            let source = URL(filePath: try Self.argument("BUNDLE"), directoryHint: .isDirectory)
            let fixture = URL(filePath: try Self.argument("WAVS"), directoryHint: .isDirectory)
            let output = URL(filePath: try Self.argument("OUTPUT"), directoryHint: .isDirectory)
            let parent = output.deletingLastPathComponent().resolvingSymlinksInPath()
            for area in [source, fixture] {
                var relation = FileManager.URLRelationship.other
                try FileManager.default.getRelationship(
                    &relation, ofDirectoryAt: area.resolvingSymlinksInPath(), toItemAt: parent)
                try #require(relation == .other)
            }
            guard mkdir(output.path, S_IRWXU) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            let file = output.appending(path: "report.json")
            let capture = VoxCPMWorkflowCapture()
            let runtime = VoxCPMRuntime(dependencies: capture.dependencies())
            let api: any SpeechGenerating = runtime
            var report = Report()
            try Self.save(report, to: file)
            do {
                let plan = try Self.plan()
                try Self.identity(of: source, report: &report)
                let voices = try await Self.readVoices(at: fixture, plan: plan)
                report.referenceManifestSHA256 = plan.referenceWAVManifestSHA256
                report.upstreamReportSHA256 = plan.pytorchGoldenReportSHA256
                let opened = try await api.open(source)
                try #require(opened.supportsReferenceAudio && capture.created == 0)
                for sample in plan.cases {
                    let ref = voices[sample.id]
                    try #require((ref == nil) == (sample.id == "none"))
                    let row = try await Self.take(
                        sample, plan: plan, voice: ref, runtime: runtime, capture: capture, output: output)
                    report.waves.append(row)
                    try Self.save(report, to: file)
                }
                try #require(report.waves.first?.loadSeconds != nil)
                let warm = report.waves.dropFirst().allSatisfy { $0.loadSeconds == nil && $0.modelSerial == 1 }
                try #require(warm && capture.created == 1)
                let recoveryCase = plan.cases[1]
                let voice = try #require(voices[recoveryCase.id])
                for fault in [VoxCPMWorkflowCapture.Fault.failureAfterEncoder, .cancelAfterEncoder] {
                    let label = try await Self.inject(
                        fault, runtime: runtime, capture: capture, voice: voice, plan: plan)
                    report.faults.append(label)
                    let row = try await Self.take(
                        recoveryCase, plan: plan, voice: voice, runtime: runtime, capture: capture, output: nil)
                    try #require(row.loadSeconds != nil && row.modelSerial == report.faults.count + 1)
                    report.recovery.append(row)
                    try Self.save(report, to: file)
                }
                await api.release()
                report.released = capture.released
                report.complete = true
                report.passed =
                    report.waves.map(\.id) == Self.ids && Set(report.waves.map(\.wavSHA256)).count == 5
                    && report.recovery.count == 2 && capture.created == 3 && report.released == [1, 2, 3]
                    && capture.injections == [.failureAfterEncoder, .cancelAfterEncoder]
            } catch {
                await api.release()
                report.released = capture.released
                report.errorType = String(describing: type(of: error))
                try Self.save(report, to: file)
                throw error
            }
            try Self.save(report, to: file)
            #expect(report.passed)
        }

        private static func argument(_ suffix: String) throws -> String {
            let value = try #require(ProcessInfo.processInfo.environment[prefix + suffix])
            try #require(!value.isEmpty)
            return value
        }

        private static func digest(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }

        private static func save(_ report: Report, to file: URL) throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: file, options: .atomic)
        }

        private static func plan() throws -> Plan {
            let url = try #require(
                Bundle.module.url(
                    forResource: "voxcpm2-reference-runtime-plan", withExtension: "json", subdirectory: "Resources"))
            let data = try Data(contentsOf: url)
            try #require(digest(data) == planSHA)
            let plan = try JSONDecoder().decode(Plan.self, from: data)
            try #require(plan.format == 1 && plan.seed == 1708 && plan.cases.map(\.id) == ids)
            return plan
        }

        private static func identity(of source: URL, report: inout Report) throws {
            let inventory = try Data(contentsOf: URL(filePath: try argument("INVENTORY")))
            let inventoryHash = digest(inventory)
            let sealed = try argument("INVENTORY_SHA256")
            try #require(inventoryHash == sealed)
            let manifest = try Data(contentsOf: source.appending(path: VoxCPMBundleInventory.manifestPath))
            let profile = try VoxCPMBundleInventory.profile(in: manifest)
            try #require(profile.rawValue == "reference_fp32")
            let files = try VoxCPMBundleInventory.files(inventory: inventory)
            let trusted = try VoxCPMBundleInventory.files(profile: profile)
            try #require(files == trusted && files.count == 25)
            let item = try #require(files.first { $0.path == VoxCPMBundleInventory.manifestPath })
            try #require(item.sha256 == digest(manifest) && item.bytes == manifest.count)
            report.manifestSHA256 = item.sha256
            report.inventorySHA256 = inventoryHash
        }

        private static func readVoices(at root: URL, plan: Plan) async throws -> [String: SpeechReferenceAudio] {
            let manifest = try Data(contentsOf: root.appending(path: "manifest.json"))
            try #require(digest(manifest) == plan.referenceWAVManifestSHA256)
            let cases = try JSONDecoder().decode(Voices.self, from: manifest)
            try #require(cases.format == 1 && cases.cases.map(\.id) == Array(ids.dropFirst()))
            let reader = SpeechReferenceReader(format: .voxcpm2)
            var voices: [String: SpeechReferenceAudio] = [:]
            for entry in cases.cases {
                try #require(entry.file == entry.id + ".wav" && entry.rate == 16_000)
                let url = root.appending(path: entry.file)
                let data = try Data(contentsOf: url)
                try #require(data.count == entry.bytes && digest(data) == entry.sha256)
                let voice = try await reader.read(url, source: .file(entry.file))
                let pcmHash = voice.samples.withUnsafeBytes { digest(Data($0)) }
                try #require(
                    voice.format == .voxcpm2 && voice.samples.count == entry.samples && pcmHash == entry.pcmSHA256)
                voices[entry.id] = voice
            }
            return voices
        }

        private static func take(
            _ sample: Plan.Sample, plan: Plan, voice: SpeechReferenceAudio?, runtime: VoxCPMRuntime,
            capture: VoxCPMWorkflowCapture, output: URL?
        ) async throws -> Wave {
            let api: any SpeechGenerating = runtime
            let old = capture.stops.count
            let beforeEncoder = capture.encoderCalls
            let audio = try await api.generate(text: plan.text, seed: plan.seed, reference: voice)
            let encoderCalls = capture.encoderCalls - beforeEncoder
            try #require(encoderCalls == (voice == nil ? 0 : 1))
            let stats = capture.stops
            try #require(stats.count == old + 1 && audio.normalizedText == plan.text)
            let stop = try #require(stats.last)
            try #require(stop.rate == 48_000 && stop.patches >= 4 && stop.stoppedPatch == stop.patches - 1)
            try #require(stop.samples == stop.patches * 7680 && audio.audioSeconds == Double(stop.samples) / 48_000)
            if let output {
                let wav = output.appending(path: sample.id + ".wav")
                try audio.wav.write(to: wav, options: .withoutOverwriting)
                let file = try AVAudioFile(forReading: wav)
                try #require(file.fileFormat.commonFormat == .pcmFormatFloat32 && file.fileFormat.channelCount == 1)
                try #require(file.length == stop.samples && file.fileFormat.sampleRate == 48_000)
            }
            return Wave(
                id: sample.id, upstreamPatches: sample.patches, stopReason: "head", stoppedPatch: stop.stoppedPatch,
                patches: stop.patches, sampleRate: stop.rate, samples: stop.samples, audioSeconds: audio.audioSeconds,
                elapsedSeconds: audio.elapsedSeconds, loadSeconds: await runtime.lastLoadSeconds,
                modelSerial: stop.serial, referenceEncoderCalls: encoderCalls, wavSHA256: digest(audio.wav))
        }

        private static func inject(
            _ fault: VoxCPMWorkflowCapture.Fault, runtime: VoxCPMRuntime, capture: VoxCPMWorkflowCapture,
            voice: SpeechReferenceAudio, plan: Plan
        ) async throws -> String {
            let api: any SpeechGenerating = runtime
            let before = capture.stops.count
            let beforeEncoder = capture.encoderCalls
            let serial = capture.created
            capture.arm(fault)
            let trial = await withTaskGroup(of: Result<Bool, any Error>.self) { group in
                group.addTask {
                    do {
                        _ = try await api.generate(text: plan.text, seed: plan.seed, reference: voice)
                        return .success(true)
                    } catch {
                        return .failure(error)
                    }
                }
                return await group.next()
            }
            guard case .failure(let error)? = trial else { throw SpeechError.invalidSamples }
            switch fault {
            case .failureAfterEncoder: try #require(error as? VoxCPMWorkflowCapture.Fault == .failureAfterEncoder)
            case .cancelAfterEncoder: try #require(error is CancellationError)
            case .none: throw SpeechError.invalidSamples
            }
            try #require(capture.stops.count == before && capture.released.last == serial)
            try #require(capture.encoderCalls == beforeEncoder + 1)
            return fault == .failureAfterEncoder ? "injected_failure_after_refenc" : "cancelled_after_refenc"
        }
    }
#endif
