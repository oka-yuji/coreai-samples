import Foundation

public struct SpeechModel: Equatable, Sendable {
    public let name: String
    public let directory: URL
    public init(name: String, directory: URL) {
        self.name = name
        self.directory = directory
    }
}

public struct SpeechAudio: Equatable, Sendable {
    public let wav: Data
    public let audioSeconds: Double
    public let elapsedSeconds: Double
    public let normalizedText: String
    public init(wav: Data, audioSeconds: Double, elapsedSeconds: Double, normalizedText: String) {
        self.wav = wav
        self.audioSeconds = audioSeconds
        self.elapsedSeconds = elapsedSeconds
        self.normalizedText = normalizedText
    }
}

public enum SpeechError: Error, Equatable, LocalizedError, Sendable {
    case noModel, busy, emptyText, invalidSeed, invalidDuration, invalidSamples
    case textTooLong(Int)
    case invalidAsset(String)
    case playbackFailed

    public var errorDescription: String? {
        switch self {
        case .noModel: "Open an Irodori model folder first."
        case .busy: "Wait for the current speech operation to finish."
        case .emptyText: "Enter some text to speak."
        case .invalidSeed: "Enter a seed from 0 to 18446744073709551615."
        case .invalidDuration: "The model returned an invalid audio duration."
        case .invalidSamples: "The model returned invalid audio samples."
        case .textTooLong(let count): "This text contains \(count) tokens. Shorten it to 255 tokens or fewer."
        case .invalidAsset(let message): message
        case .playbackFailed: "The generated audio could not be played."
        }
    }
}
