import AVFAudio
import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

struct SpeechReferenceIntegrationTests {
    private struct Fixture: Decodable {
        let text: String
        let seed: UInt64
        let reference: String
        let expected: String
        let frames: Int
    }

    @Test(
        "REF-CORE-04 Native reference generation records waveform fidelity and preserves finite matching frames",
        .enabled(
            if: ProcessInfo.processInfo.environment["COREAI_IRODORI_REFERENCE_BUNDLE"] != nil
                && ProcessInfo.processInfo.environment["COREAI_IRODORI_REFERENCE_FIXTURES"] != nil))
    func nativeGeneration() async throws {
        let environment = ProcessInfo.processInfo.environment
        let bundle = try #require(environment["COREAI_IRODORI_REFERENCE_BUNDLE"])
        let fixtures = URL(filePath: try #require(environment["COREAI_IRODORI_REFERENCE_FIXTURES"]))
        let report = URL(filePath: try #require(environment["COREAI_IRODORI_REFERENCE_REPORT"]))
        let runtime = IrodoriRuntime()
        do {
            let model = try await runtime.open(URL(filePath: bundle))
            #expect(model.supportsReferenceAudio)
            var results: [[String: Any]] = []
            for index in 0..<3 {
                let fixture = try JSONDecoder().decode(
                    Fixture.self, from: Data(contentsOf: fixtures.appending(path: "case-\(index).json")))
                let reference = try SpeechReferenceAudio(
                    samples: samples(Data(contentsOf: fixtures.appending(path: fixture.reference))),
                    source: .file(fixture.reference))
                let expected = try samples(Data(contentsOf: fixtures.appending(path: fixture.expected)))
                let audio = try await runtime.generate(text: fixture.text, seed: fixture.seed, reference: reference)
                let actual = try samples(Data(audio.wav.dropFirst(58)))
                #expect(actual.count == fixture.frames * 1920)
                #expect(actual.allSatisfy { $0.isFinite })
                let metrics = try comparison(expected, actual)
                results.append([
                    "case": index, "referenceSamples": reference.samples.count,
                    "frames": actual.count / 1920, "elapsedSeconds": audio.elapsedSeconds,
                    "nrmse": metrics.nrmse, "cosine": metrics.cosine,
                    "originalExploratoryGate": metrics.nrmse <= 0.001 && metrics.cosine >= 0.99999,
                ])
                print("Native reference case \(index): NRMSE \(metrics.nrmse), cosine \(metrics.cosine)")
            }
            let fixture = try JSONDecoder().decode(
                Fixture.self, from: Data(contentsOf: fixtures.appending(path: "case-0.json")))
            let unconditioned = try await runtime.generate(text: fixture.text, seed: fixture.seed)
            #expect(unconditioned.audioSeconds > 0 && unconditioned.wav.count > 58)
            try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
                .write(to: report, options: .withoutOverwriting)
            await runtime.release()
        } catch {
            await runtime.release()
            throw error
        }
    }

    @Test(
        "REF-CORE-04 Native generation accepts a real reference recording and remains stable across voice changes",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_IRODORI_SMOKE_REFERENCE"] != nil))
    func nativeSmoke() async throws {
        let environment = ProcessInfo.processInfo.environment
        let bundle = try #require(environment["COREAI_IRODORI_REFERENCE_BUNDLE"])
        let source = URL(filePath: try #require(environment["COREAI_IRODORI_SMOKE_REFERENCE"]))
        let audioFile = try AVAudioFile(forReading: source, commonFormat: .pcmFormatFloat32, interleaved: false)
        #expect(audioFile.processingFormat.sampleRate == 48_000 && audioFile.processingFormat.channelCount == 1)
        let buffer = try #require(
            AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: UInt32(audioFile.length)))
        try audioFile.read(into: buffer)
        let channels = try #require(buffer.floatChannelData)
        let reference = try SpeechReferenceAudio(
            samples: Array(UnsafeBufferPointer(start: channels[0], count: Int(buffer.frameLength))),
            source: .file(source.lastPathComponent))
        let textURL = try #require(
            Bundle.module.url(forResource: "reference-input", withExtension: "json", subdirectory: "Resources"))
        let texts = try JSONDecoder().decode([String].self, from: Data(contentsOf: textURL))
        let text = try #require(texts.first)
        let runtime = IrodoriRuntime()
        do {
            _ = try await runtime.open(URL(filePath: bundle))
            let first = try await runtime.generate(text: text, seed: 1708, reference: reference)
            let unconditioned = try await runtime.generate(text: text, seed: 1708)
            let repeated = try await runtime.generate(text: text, seed: 1708, reference: reference)
            #expect(first.wav != unconditioned.wav)
            #expect(first.wav == repeated.wav)
            #expect(try samples(Data(first.wav.dropFirst(58))).allSatisfy { $0.isFinite })
            for count in [SpeechReferenceAudio.minimumSamples, SpeechReferenceAudio.maximumSamples] {
                let repeatedSamples = (0..<count).map { reference.samples[$0 % reference.samples.count] }
                let boundary = try SpeechReferenceAudio(samples: repeatedSamples, source: .recording)
                let output = try await runtime.generate(text: text, seed: 1708, reference: boundary)
                #expect(output.audioSeconds > 0 && output.audioSeconds <= 30)
                #expect(try samples(Data(output.wav.dropFirst(58))).allSatisfy { $0.isFinite })
                print("Native reference boundary: \(boundary.seconds)s input, \(output.audioSeconds)s output")
            }
            if let directory = environment["COREAI_IRODORI_SMOKE_OUTPUT"] {
                let root = URL(filePath: directory)
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                try first.wav.write(to: root.appending(path: "reference.wav"), options: .withoutOverwriting)
                try unconditioned.wav.write(to: root.appending(path: "unconditioned.wav"), options: .withoutOverwriting)
            }
            print(
                "Native reference smoke: \(reference.seconds)s input, \(first.audioSeconds)s output, \(first.elapsedSeconds)s generation"
            )
            await runtime.release()
        } catch {
            await runtime.release()
            throw error
        }
    }

    private func samples(_ data: Data) throws -> [Float] {
        guard data.count.isMultiple(of: 4) else { throw SpeechError.invalidSamples }
        return data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
            }
        }
    }

    private func comparison(_ expected: [Float], _ actual: [Float]) throws -> (nrmse: Double, cosine: Double) {
        guard expected.count == actual.count, !expected.isEmpty else { throw SpeechError.invalidSamples }
        var squaredError = 0.0
        var expectedPower = 0.0
        var actualPower = 0.0
        var dot = 0.0
        for (reference, output) in zip(expected, actual) {
            let reference = Double(reference)
            let output = Double(output)
            squaredError += (reference - output) * (reference - output)
            expectedPower += reference * reference
            actualPower += output * output
            dot += reference * output
        }
        guard expectedPower > 0, actualPower > 0 else { throw SpeechError.invalidSamples }
        return (sqrt(squaredError / expectedPower), dot / sqrt(expectedPower * actualPower))
    }
}
