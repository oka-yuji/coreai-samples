#if os(macOS)
    import CoreAI
    import CryptoKit
    import Darwin
    import Foundation
    import SpeechAssets
    import Testing

    @testable import SpeechRuntime

    @Suite(.serialized)
    struct VoxCPMReferenceNativeTests {
        private struct FileEntry: Decodable {
            let file: String
            let bytes: Int
            let sha256: String
        }

        private struct VoiceCase: Decodable {
            let id: String
            let samples: Int
            let patches: Int
            let pcm: FileEntry
            let original: FileEntry
        }

        private struct Index: Decodable {
            let format: Int
            let modelRevision: String
            let upstreamRevision: String
            let precision: String
            let limit: Double
            let cases: [VoiceCase]
        }

        private struct Row: Encodable {
            let id: String
            let samples: Int
            let patches: Int
            let compared: Int
            let nonfinite: Int
            let maxAbsolute: Double?
            let normalizedMax: Double?
            let passed: Bool
            let seconds: Double
        }

        private struct Report: Encodable {
            let format = 1
            let criterion = 0.001
            let fixtureSHA256 = VoxCPMReferenceNativeTests.fixtureDigest
            var complete = false
            var passed = false
            var inventorySHA256: String?
            var manifestSHA256: String?
            var files = 0
            var bytes: Int64 = 0
            var models = 0
            var functions = [String]()
            var loadSeconds: Double?
            var rows = [Row]()
            var errorType: String?
        }

        private static let prefix = "COREAI_VOXCPM2_REFENC_"
        private static let enabled = ProcessInfo.processInfo.environment[prefix + "BUNDLE"] != nil
        private static let fixtureDigest = "91bae4da12ca98bc583bd5a365114fd9ce8e2ff8bed543ec317e545ccfacf20e"
        private static let order = ["tslm", "ralm", "proj", "locenc", "dit", "vae", "refenc"]
        private static let ids = ["0a67-5", "0a67-10", "1410-5", "1410-10"]

        @Test(
            "VRA-04 The packaged FP32 reference encoder matches all four original feature tensors",
            .enabled(if: enabled))
        func packagedEncoder() async throws {
            let output = URL(filePath: try Self.argument("OUTPUT"), directoryHint: .isDirectory)
            let source = URL(filePath: try Self.argument("BUNDLE"), directoryHint: .isDirectory)
            let fixtures = URL(filePath: try Self.argument("FIXTURES"), directoryHint: .isDirectory)
            let parent = output.deletingLastPathComponent().resolvingSymlinksInPath()
            for area in [source, fixtures] {
                var relation = FileManager.URLRelationship.other
                try FileManager.default.getRelationship(
                    &relation, ofDirectoryAt: area.resolvingSymlinksInPath(), toItemAt: parent)
                try #require(relation == .other)
            }
            guard mkdir(output.path, S_IRWXU) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            let file = output.appending(path: "report.json")
            var report = Report()
            try Self.save(report, to: file)
            let copies = output.appending(path: "verified", directoryHint: .isDirectory)
            defer { try? FileManager.default.removeItem(at: copies) }
            do {
                let index = try Self.index(at: fixtures)
                try Self.sourceIdentity(source, report: &report)
                try Self.save(report, to: file)
                let loaded = try await VoxCPMPipeline<VoxCPMAssets>.load(bundle: source, copyingInto: copies)
                report.loadSeconds = loaded.statistics.totalSeconds
                let assets = loaded.pipeline.components
                let kind = try Self.checkLoaded(loaded.pipeline, urls: loaded.bundle.assets)
                report.models = assets.models.count
                report.functions = assets.functions.keys.map(\.rawValue).sorted()
                for voice in index.cases {
                    report.rows.append(try await Self.compare(voice, in: fixtures, assets: assets, kind: kind))
                    try Self.save(report, to: file)
                }
                report.complete = true
                report.passed =
                    report.rows.map(\.id) == Self.ids && report.rows.allSatisfy(\.passed)
                    && report.rows.map(\.compared).reduce(0, +) == 48_640
            } catch {
                report.errorType = String(describing: type(of: error))
                try Self.save(report, to: file)
                throw error
            }
            try Self.save(report, to: file)
            #expect(report.passed)
        }

        private static func argument(_ name: String) throws -> String {
            let value = try #require(ProcessInfo.processInfo.environment[prefix + name])
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

        private static func index(at root: URL) throws -> Index {
            let data = try Data(contentsOf: root.appending(path: "cases.json"))
            try #require(digest(data) == fixtureDigest)
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let index = try decoder.decode(Index.self, from: data)
            try #require(index.format == 1 && index.precision == "float32_le" && index.limit == 0.001)
            try #require(index.modelRevision == VoxCPMBundleInventory.revision)
            try #require(index.upstreamRevision == VoxCPMBundleInventory.upstreamRevision)
            try #require(index.cases.map(\.id) == ids)
            try #require(index.cases.map(\.samples) == [80_000, 160_000, 80_000, 160_000])
            try #require(index.cases.map(\.patches) == [32, 63, 32, 63])
            return index
        }

        private static func sourceIdentity(_ root: URL, report: inout Report) throws {
            let inventory = try Data(contentsOf: URL(filePath: try argument("INVENTORY")))
            let inventoryHash = digest(inventory)
            let selectedHash = try argument("INVENTORY_SHA256")
            try #require(inventoryHash == selectedHash)
            let raw = try Data(contentsOf: root.appending(path: VoxCPMBundleInventory.manifestPath))
            let profile = try VoxCPMBundleInventory.profile(in: raw)
            try #require(profile.rawValue == "reference_fp32")
            let files = try VoxCPMBundleInventory.files(inventory: inventory)
            let trusted = try VoxCPMBundleInventory.files(profile: profile)
            try #require(files == trusted && files.count == 25)
            let manifest = try #require(files.first { $0.path == VoxCPMBundleInventory.manifestPath })
            try #require(digest(raw) == manifest.sha256 && raw.count == manifest.bytes)
            report.manifestSHA256 = manifest.sha256
            report.inventorySHA256 = inventoryHash
            report.files = files.count
            report.bytes = files.map(\.bytes).reduce(0, +)
        }

        private static func checkLoaded(_ pipeline: VoxCPMPipeline<VoxCPMAssets>, urls: [URL]) throws -> VoxCPMFunction
        {
            let assets = pipeline.components
            try #require(pipeline.contract.profile.rawValue == "reference_fp32")
            try #require(assets.models.count == 7 && assets.functions.count == 12)
            try #require(urls.map(\.lastPathComponent) == order.map { $0 + "-fp32.aimodel" })
            for (name, model) in zip(order, assets.models) {
                var observed: [String: VoxCPMSignature] = [:]
                for functionName in model.functionNames {
                    let descriptor = try #require(model.functionDescriptor(for: functionName))
                    observed[functionName] = VoxCPMSignature(descriptor)
                }
                try pipeline.contract.check(asset: name, functionNames: model.functionNames, observed: observed)
            }
            let kind = try #require(VoxCPMFunction(rawValue: "refenc.main"))
            let signature = try #require(assets.signatures[kind])
            try #require(signature.inputs == [.init(name: "pcm", type: "float32", shape: [1, 1, 161_280])])
            try #require(signature.outputs == [.init(name: "mu", type: "float32", shape: [1, 64, 252])])
            try #require(signature.states.isEmpty)
            return kind
        }

        private static func values(_ entry: FileEntry, name: String, count: Int, root: URL) throws -> [Float] {
            try #require(entry.file == name && entry.bytes == count * 4)
            let data = try Data(contentsOf: root.appending(path: name))
            try #require(data.count == entry.bytes && digest(data) == entry.sha256)
            let result = data.withUnsafeBytes { raw in
                (0..<count).map { index in
                    Float(
                        bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)))
                }
            }
            let finite = result.allSatisfy(\.isFinite)
            try #require(finite)
            return result
        }

        private static func compare(_ voice: VoiceCase, in root: URL, assets: VoxCPMAssets, kind: VoxCPMFunction)
            async throws -> Row
        {
            var pcm = try values(voice.pcm, name: voice.id + ".pcm.f32", count: voice.samples, root: root)
            let reference = try values(
                voice.original, name: voice.id + ".original.f32", count: voice.patches * 4 * 64, root: root)
            pcm.append(contentsOf: repeatElement(0, count: 161_280 - pcm.count))
            var states = try assets.makeStates()
            let start = ContinuousClock.now
            let result = try await assets.call(kind, ["pcm": .floats(pcm)], states: &states)
            let seconds = (ContinuousClock.now - start) / .seconds(1)
            let mu = try #require(result["mu"])
            try #require(mu.count == 64 * 252 && result.count == 1)
            let nonfinite = mu.count { !$0.isFinite }
            var worst = 0.0
            var peak = 0.0
            for (index, original) in reference.enumerated() {
                let actual = mu[(index % 64) * 252 + index / 64]
                worst = max(worst, abs(Double(actual) - Double(original)))
                peak = max(peak, abs(Double(original)))
            }
            let normalized = worst / max(peak, 1e-8)
            return Row(
                id: voice.id, samples: voice.samples, patches: voice.patches, compared: reference.count,
                nonfinite: nonfinite, maxAbsolute: nonfinite == 0 ? worst : nil,
                normalizedMax: nonfinite == 0 ? normalized : nil,
                passed: nonfinite == 0 && normalized <= 0.001, seconds: seconds)
        }
    }
#endif
