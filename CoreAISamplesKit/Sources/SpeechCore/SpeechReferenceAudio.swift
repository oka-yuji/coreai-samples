import Foundation

public enum SpeechReferenceSource: Equatable, Sendable {
    case recording
    case file(String)
}

public struct SpeechReferenceAudio: Equatable, Sendable {
    public static let sampleRate = 48_000
    public static let minimumSamples = 24_000
    public static let maximumSamples = 1_440_000
    public let samples: [Float]
    public let source: SpeechReferenceSource

    public init(samples: [Float], source: SpeechReferenceSource) throws {
        guard samples.count >= Self.minimumSamples else { throw SpeechReferenceError.tooShort }
        guard samples.count <= Self.maximumSamples else { throw SpeechReferenceError.tooLong }
        guard samples.allSatisfy({ $0.isFinite }) else { throw SpeechReferenceError.invalidSamples }
        let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
        guard peak > 0 else { throw SpeechReferenceError.silent }
        self.samples = peak > 1 ? samples.map { $0 / peak } : samples
        self.source = source
    }

    public var seconds: Double { Double(samples.count) / Double(Self.sampleRate) }
    public var frameCount: Int { (samples.count + 1919) / 1920 }
    public var speakerTokenCount: Int { frameCount / 4 + 1 }

    public var encoderSamples: [Float] {
        var result = samples
        let padding = frameCount * 1920 - samples.count
        for offset in 0..<padding { result.append(samples[samples.count - 2 - offset]) }
        return result
    }
}

public enum SpeechReferenceError: Error, Equatable, LocalizedError, Sendable {
    case tooShort, tooLong, invalidSamples, silent, unreadable, permissionDenied, recordingFailed

    public var errorDescription: String? {
        switch self {
        case .tooShort: "Use a voice recording of at least half a second."
        case .tooLong: "Choose a voice recording no longer than 30 seconds."
        case .invalidSamples: "The reference audio contains invalid samples."
        case .silent: "The recording is silent. Record your voice and try again."
        case .unreadable: "This audio file could not be read. Choose a WAV, M4A, MP3 or AIFF file."
        case .permissionDenied: "Allow microphone access in Settings to record a reference voice."
        case .recordingFailed: "Recording stopped unexpectedly. Please try again."
        }
    }
}
