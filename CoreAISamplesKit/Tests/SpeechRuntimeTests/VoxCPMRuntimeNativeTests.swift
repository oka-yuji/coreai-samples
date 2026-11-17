import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

@Suite(.serialized)
struct VoxCPMRuntimeNativeTests {
    struct Smoke: Decodable {
        let seed: UInt64
        let plain: String
        let design: String
    }

    static let environment = ProcessInfo.processInfo.environment
    static let enabled = ["COREAI_VOXCPM2_RUNTIME_BUNDLE", "COREAI_VOXCPM2_RUNTIME_OUTPUT"].allSatisfy {
        environment[$0] != nil
    }

    @Test(
        "VRT-07 A real bundle opens, generates twice with one load, releases and generates again after a reload",
        .enabled(if: enabled))
    func smoke() async throws {
        let bundle = URL(
            filePath: try #require(Self.environment["COREAI_VOXCPM2_RUNTIME_BUNDLE"]), directoryHint: .isDirectory)
        let output = URL(
            filePath: try #require(Self.environment["COREAI_VOXCPM2_RUNTIME_OUTPUT"]), directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let resource = try #require(
            Bundle.module.url(forResource: "voxcpm2-runtime-smoke", withExtension: "json", subdirectory: "Resources"))
        let smoke = try JSONDecoder().decode(Smoke.self, from: Data(contentsOf: resource))
        let runtime = VoxCPMRuntime()
        let clock = ContinuousClock()
        var wavs: [String: Data] = [:]
        func generate(_ name: String, _ text: String) async throws -> (
            load: Double?, memory: VoxCPMMemory, json: [String: Any]
        ) {
            let audio = try await runtime.generate(text: text, seed: smoke.seed)
            let load = await runtime.lastLoadSeconds
            let memory = VoxCPMMemory.capture()
            try audio.wav.write(to: output.appending(path: name + ".wav"), options: .withoutOverwriting)
            wavs[name] = audio.wav
            return (
                load, memory,
                [
                    "name": name, "audioSeconds": audio.audioSeconds, "elapsedSeconds": audio.elapsedSeconds,
                    "loadSeconds": load.map { $0 as Any } ?? NSNull(),
                    "generationSeconds": audio.elapsedSeconds - (load ?? 0), "wavBytes": audio.wav.count,
                    "memoryAfter": memory.json,
                ]
            )
        }
        func halved(_ loaded: VoxCPMMemory, _ released: VoxCPMMemory) -> Bool {
            guard let loaded = loaded.residentBytes, let released = released.residentBytes else { return false }
            return released < loaded / 2
        }
        var record: [String: Any] = [
            "bundle": bundle.lastPathComponent, "seed": smoke.seed,
            "system": ProcessInfo.processInfo.operatingSystemVersionString,
            "memoryBeforeOpen": VoxCPMMemory.capture().json,
        ]
        var start = clock.now
        let model = try await runtime.open(bundle)
        record["model"] = model.name
        record["openSeconds"] = (clock.now - start) / .seconds(1)
        record["memoryAfterOpen"] = VoxCPMMemory.capture().json
        let first = try await generate("first", smoke.plain)
        let reused = try await generate("reused", smoke.design)
        start = clock.now
        await runtime.release()
        record["releaseSeconds"] = (clock.now - start) / .seconds(1)
        let released = VoxCPMMemory.capture()
        record["memoryAfterRelease"] = released.json
        let reloaded = try await generate("reloaded", smoke.plain)
        start = clock.now
        await runtime.release()
        record["finalReleaseSeconds"] = (clock.now - start) / .seconds(1)
        let finallyReleased = VoxCPMMemory.capture()
        record["memoryAfterFinalRelease"] = finallyReleased.json
        record["steps"] = [first.json, reused.json, reloaded.json]
        record["reloadedMatchesFirst"] = wavs["first"] == wavs["reloaded"]
        try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appending(path: "runtime-smoke.json"), options: .withoutOverwriting)
        #expect(first.load != nil)
        #expect(reused.load == nil)
        #expect(reloaded.load != nil)
        #expect(halved(reused.memory, released))
        #expect(halved(reloaded.memory, finallyReleased))
    }
}
