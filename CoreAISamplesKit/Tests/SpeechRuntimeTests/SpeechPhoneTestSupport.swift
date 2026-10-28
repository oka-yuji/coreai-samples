import Foundation
import SpeechCore
import SpeechFeature

@testable import SpeechRuntime

func phoneTestDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

@MainActor
final class FakePhoneEnvironment: PhoneCheckEnvironment {
    var begins = 0
    var ends = 0
    func begin() { begins += 1 }
    func end() { ends += 1 }
    func memory() -> PhoneMemory {
        PhoneMemory(availableBytes: 100, footprintBytes: 20, peakFootprintBytes: 30, thermalState: 0)
    }
    func info() -> PhoneDeviceInfo {
        PhoneDeviceInfo(
            osVersion: "fake", osBuild: "fake", hardware: "fake", batteryState: 0,
            lowPowerMode: false, architecture: "fake")
    }
}

@MainActor
final class FakeCheckSpeechService: SpeechService {
    var failure = ""
    var block = false
    let started = TestSignal()
    let stopCalled = TestSignal()
    let gate = TestGate()
    var opens = 0
    var seeds: [UInt64] = []
    var texts: [String] = []
    var plays = 0
    var stops = 0
    var referenceToImport: SpeechReferenceAudio?
    var generatedReferences: [SpeechReferenceAudio?] = []
    func open(path: String) throws -> SpeechModel {
        opens += 1
        if failure == "open" { throw SpeechError.invalidSamples }
        return SpeechModel(name: "Fixture", directory: URL(filePath: path))
    }
    func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
        texts.append(text)
        seeds.append(seed)
        started.fire()
        if block { await gate.wait() }
        if failure == "generate" { throw SpeechError.invalidSamples }
        return SpeechAudio(wav: Data([1, 2]), audioSeconds: 1, elapsedSeconds: 2, normalizedText: text)
    }
    func play(_ data: Data) throws {
        plays += 1
        if failure == "play" { throw SpeechError.invalidSamples }
        if failure == "interrupt" { throw CancellationError() }
    }
    func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
        generatedReferences.append(reference)
        return try await generate(text: text, seed: seed)
    }
    func importReference(_ url: URL) throws -> SpeechReferenceAudio {
        guard let referenceToImport else { throw SpeechReferenceError.unreadable }
        return referenceToImport
    }
    func stop() {
        stops += 1
        stopCalled.fire()
    }
}

actor FakeFolderListing: SpeechFolderListing {
    let started = TestSignal()
    let gate = TestGate()
    var fails = false
    func configure(fails: Bool) { self.fails = fails }
    func folders() async throws -> [SpeechFolder] {
        started.fire()
        await gate.wait()
        if fails { throw SpeechError.invalidSamples }
        return [SpeechFolder(url: URL(filePath: "/fixture"))]
    }
}
