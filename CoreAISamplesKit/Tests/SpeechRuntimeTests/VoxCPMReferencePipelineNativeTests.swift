#if os(macOS)
    import CryptoKit
    import Darwin
    import Foundation
    import SpeechAssets
    import SpeechCore
    import Testing

    @testable import SpeechRuntime

    @Suite(.serialized)
    struct VoxCPMReferencePipelineNativeTests {
        private struct Entry: Decodable {
            let file: String
            let bytes: Int
            let sha256: String
        }
        private struct Case: Decodable {
            let id: String
            let target: [Int32]
            let length: Int
            let patches: Int
            let originalStop: Int
            let files: [String: Entry]
        }
        private struct Index: Decodable {
            let format: Int
            let seed: UInt64
            let limit: Double
            let cases: [Case]
        }
        private struct Wave: Encodable {
            let id: String
            let sourceStop: Int
            let stoppedPatch: Int
            let patches: Int
            let samples: Int
            let sampleRate: Int
            let seconds: Double
            let wavSHA256: String
        }
        private struct Report: Encodable {
            let format = 1
            let criterion = 0.001
            let freeTimingIncludesTraceWrites = true
            let fixtureSHA256 = VoxCPMReferencePipelineNativeTests.fixtureSHA
            var inventorySHA256: String?
            var manifestSHA256: String?
            var verifiedCopyKey: String?
            var complete = false
            var freeGenerationPassed = false
            var sameInputCalls = 0
            var waves = [Wave]()
            var errorType: String?
        }
        private static let prefix = "COREAI_VOXCPM2_PREFIX_"
        private static let enabled = ProcessInfo.processInfo.environment[prefix + "BUNDLE"] != nil
        private static let fixtureSHA = "3629a3d2bd16a382ab51de96485fade846392cf458a2854afdf518c29d87d396"
        private static let ids = ["none", "0a67-5", "0a67-10", "1410-5", "1410-10"]

        @Test(
            "VRP-04 The direct reference pipeline saves five free generations, real prefix traces and same-input prefills",
            .enabled(if: enabled))
        func directFive() async throws {
            let source = URL(filePath: try Self.arg("BUNDLE"), directoryHint: .isDirectory)
            let fixtures = URL(filePath: try Self.arg("FIXTURES"), directoryHint: .isDirectory)
            let output = URL(filePath: try Self.arg("OUTPUT"), directoryHint: .isDirectory)
            let parent = output.deletingLastPathComponent().resolvingSymlinksInPath()
            for area in [source, fixtures] {
                var relation = FileManager.URLRelationship.other
                try FileManager.default.getRelationship(
                    &relation, ofDirectoryAt: area.resolvingSymlinksInPath(), toItemAt: parent)
                try #require(relation == .other)
            }
            guard mkdir(output.path, S_IRWXU) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            let copies = output.appending(path: "verified", directoryHint: .isDirectory)
            defer { try? FileManager.default.removeItem(at: copies) }
            let resultFile = output.appending(path: "swift-report.json")
            var report = Report()
            try Self.save(report, to: resultFile)
            do {
                let evidence = try Self.identity(source, report: &report)
                let index = try Self.index(fixtures)
                let loaded = try await VoxCPMPipeline<VoxCPMAssets>.load(bundle: source, copyingInto: copies)
                let confirmed = evidence.matches(loaded.bundle, source: source, copies: copies)
                try #require(confirmed)
                report.verifiedCopyKey = evidence.key
                let real = loaded.pipeline.components
                for sample in index.cases {
                    let caseDir = output.appending(path: sample.id, directoryHint: .isDirectory)
                    let trace = try VoxCPMPrefixTrace(folder: caseDir)
                    let probe = VoxCPMPipeline(
                        components: VoxCPMPrefixTrace.Parts(physical: real, trace: trace),
                        contract: loaded.pipeline.contract)
                    let reference = try Self.reference(sample, fixtures: fixtures)
                    let start = ContinuousClock.now
                    let speech = try await probe.generate(
                        ids: sample.target, seed: index.seed, reference: reference, observer: nil)
                    let seconds = (ContinuousClock.now - start) / .seconds(1)
                    let row = try Self.saveWave(speech, sample: sample, seconds: seconds, folder: caseDir)
                    report.waves.append(row)
                    for name in ["tslm", "ralm"] {
                        try await Self.sameInput(
                            name, sample: sample, assets: real, fixtures: fixtures, folder: caseDir)
                        report.sameInputCalls += 1
                    }
                    try Self.save(report, to: resultFile)
                }
                report.complete = true
                report.freeGenerationPassed = report.waves.map(\.id) == Self.ids && report.sameInputCalls == 10
            } catch {
                report.errorType = String(describing: type(of: error))
                try Self.save(report, to: resultFile)
                throw error
            }
            try Self.save(report, to: resultFile)
            #expect(report.freeGenerationPassed)
        }

        private static func arg(_ name: String) throws -> String {
            let v = try #require(ProcessInfo.processInfo.environment[prefix + name])
            try #require(!v.isEmpty)
            return v
        }
        private static func hash(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        private static func save(_ value: Report, to file: URL) throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(value).write(to: file, options: .atomic)
        }
        private static func index(_ root: URL) throws -> Index {
            let data = try Data(contentsOf: root.appending(path: "cases.json"))
            try #require(hash(data) == fixtureSHA)
            let i = try JSONDecoder().decode(Index.self, from: data)
            try #require(i.format == 1 && i.seed == 1708 && i.limit == 0.001 && i.cases.map(\.id) == ids)
            try #require(i.cases.map(\.length) == [20, 54, 85, 54, 85])
            try #require(i.cases.map(\.patches) == [0, 32, 63, 32, 63])
            return i
        }
        private static func identity(_ source: URL, report: inout Report) throws -> VoxCPMVerifiedCopyEvidence {
            let inv = try Data(contentsOf: URL(filePath: try arg("INVENTORY")))
            let seal = try arg("INVENTORY_SHA256")
            try #require(hash(inv) == seal)
            let raw = try Data(contentsOf: source.appending(path: VoxCPMBundleInventory.manifestPath))
            let profile = try VoxCPMBundleInventory.profile(in: raw)
            try #require(profile.rawValue == "reference_fp32")
            let rows = try VoxCPMBundleInventory.files(profile: profile)
            let external = try VoxCPMBundleInventory.files(inventory: inv)
            try #require(rows == external && rows.count == 25)
            let manifest = try #require(rows.first { $0.path == VoxCPMBundleInventory.manifestPath })
            try #require(manifest.sha256 == hash(raw) && manifest.bytes == raw.count)
            report.manifestSHA256 = manifest.sha256
            report.inventorySHA256 = seal
            let layout = try SpeechBundleLayout.voxcpm2(manifest: raw)
            return VoxCPMVerifiedCopyEvidence(
                key: ValidatedSpeechBundle.verifiedKey(manifest: raw, files: rows, names: .voxcpm2),
                assetPaths: layout.assets)
        }
        private static func floats(_ key: String, sample: Case, count: Int, root: URL) throws -> [Float] {
            let entry = try #require(sample.files[key])
            try #require(entry.file == sample.id + "." + key + ".f32" && entry.bytes == count * 4)
            let raw = try Data(contentsOf: root.appending(path: entry.file))
            try #require(raw.count == entry.bytes && hash(raw) == entry.sha256)
            let values = raw.withUnsafeBytes { bytes in
                (0..<count).map {
                    Float(
                        bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)))
                }
            }
            let finite = values.allSatisfy(\.isFinite)
            try #require(finite)
            return values
        }
        private static func reference(_ sample: Case, fixtures: URL) throws -> SpeechReferenceAudio? {
            if sample.id == "none" { return nil }
            let count = sample.patches == 32 ? 80_000 : 160_000
            return try SpeechReferenceAudio(
                samples: floats("pcm", sample: sample, count: count, root: fixtures),
                source: .file(sample.id), format: .voxcpm2)
        }
        private static func sameInput(_ name: String, sample: Case, assets: VoxCPMAssets, fixtures: URL, folder: URL)
            async throws
        {
            var input = try floats(name + "_prefill_in", sample: sample, count: sample.length * 2048, root: fixtures)
            input.append(contentsOf: repeatElement(0, count: (145 - sample.length) * 2048))
            var zeroed = try assets.makeStates()
            let f: VoxCPMFunction = name == "tslm" ? .prefill : .residualPrefill
            let outputs = try await assets.call(
                f, ["embeds": .floats(input), "length": .integers([Int32(sample.length)])], states: &zeroed)
            let hidden = try #require(outputs["hidden"])
            try #require(hidden.count == 145 * 2048)
            let bytes = hidden.withUnsafeBytes { Data($0) }
            try bytes.write(to: folder.appending(path: "same-input-" + name + ".f32"), options: .withoutOverwriting)
        }
        private static func saveWave(_ speech: VoxCPMSpeech, sample: Case, seconds: Double, folder: URL) throws -> Wave
        {
            let maxPatches = min(6 * (sample.target.count - 1) + 10, 1024 - sample.length)
            try #require(
                speech.patchCount >= 4 && speech.patchCount <= maxPatches
                    && speech.stoppedPatch == speech.patchCount - 1)
            try #require(speech.sampleRate == 48_000 && speech.samples.count == speech.patchCount * 7680)
            let wav = try floatWAV(samples: speech.samples, maximumSamples: maxPatches * 7680)
            try wav.write(to: folder.appending(path: "generated.wav"), options: .withoutOverwriting)
            return Wave(
                id: sample.id, sourceStop: sample.originalStop, stoppedPatch: speech.stoppedPatch,
                patches: speech.patchCount,
                samples: speech.samples.count, sampleRate: speech.sampleRate, seconds: seconds, wavSHA256: hash(wav))
        }
    }
#endif
