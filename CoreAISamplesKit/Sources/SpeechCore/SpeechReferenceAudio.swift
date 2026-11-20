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
    public let format: SpeechReferenceFormat

    public init(samples: [Float], source: SpeechReferenceSource) throws {
        try self.init(samples: samples, source: source, format: .irodori)
    }

    public init(samples: [Float], source: SpeechReferenceSource, format: SpeechReferenceFormat) throws {
        guard samples.count >= format.minimumSamples else { throw format.tooShortError }
        guard samples.count <= format.maximumSamples else { throw format.tooLongError }
        guard samples.allSatisfy({ $0.isFinite }) else { throw SpeechReferenceError.invalidSamples }
        let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
        guard peak > 0 else { throw SpeechReferenceError.silent }
        self.samples = peak > 1 ? samples.map { $0 / peak } : samples
        self.source = source
        self.format = format
    }

    public var seconds: Double { Double(samples.count) / Double(format.sampleRate) }

    public var frameCount: Int {
        switch format {
        case .irodori: (samples.count + 1919) / 1920
        case .voxcpm2: 4 * voxcpmPatchCount
        }
    }

    public var conditioningPositionCount: Int {
        switch format {
        case .irodori: frameCount / 4 + 1
        case .voxcpm2: voxcpmPatchCount
        }
    }

    public var speakerTokenCount: Int { conditioningPositionCount }

    public var encoderSamples: [Float] {
        var result = samples
        switch format {
        case .irodori:
            let padding = frameCount * 1920 - samples.count
            for offset in 0..<padding { result.append(samples[samples.count - 2 - offset]) }
        case .voxcpm2:
            result.append(contentsOf: repeatElement(Float(0), count: Self.voxcpmEncoderSamples - samples.count))
        }
        return result
    }

    private static let voxcpmEncoderSamples = 161_280

    private var voxcpmPatchCount: Int { (samples.count + 2559) / 2560 }
}

public enum SpeechReferenceError: Error, Equatable, LocalizedError, Sendable {
    case tooShort, tooLong, invalidSamples, silent, unreadable, permissionDenied, recordingFailed
    case voxcpmTooShort, voxcpmTooLong, incompatibleFormat

    public var errorDescription: String? {
        switch self {
        case .tooShort: "Use a voice recording of at least half a second."
        case .tooLong: "Choose a voice recording no longer than 30 seconds."
        case .invalidSamples: "The reference audio contains invalid samples."
        case .silent: "The recording is silent. Record your voice and try again."
        case .unreadable: "This audio file could not be read. Choose a WAV, M4A, MP3 or AIFF file."
        case .permissionDenied: "Allow microphone access in Settings to record a reference voice."
        case .recordingFailed: "Recording stopped unexpectedly. Please try again."
        case .voxcpmTooShort: "Use a voice recording of at least 5 seconds."
        case .voxcpmTooLong: "Choose a voice recording no longer than 10 seconds."
        case .incompatibleFormat: "Record or choose the reference voice again for this model."
        }
    }
}
