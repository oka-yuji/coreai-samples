public enum SpeechReferenceFormat: Equatable, Sendable {
    case irodori
    case voxcpm2

    public var sampleRate: Int {
        switch self {
        case .irodori: 48_000
        case .voxcpm2: 16_000
        }
    }

    public var minimumSamples: Int {
        switch self {
        case .irodori: 24_000
        case .voxcpm2: 80_000
        }
    }

    public var maximumSamples: Int {
        switch self {
        case .irodori: 1_440_000
        case .voxcpm2: 160_000
        }
    }

    var tooShortError: SpeechReferenceError {
        switch self {
        case .irodori: .tooShort
        case .voxcpm2: .voxcpmTooShort
        }
    }

    var tooLongError: SpeechReferenceError {
        switch self {
        case .irodori: .tooLong
        case .voxcpm2: .voxcpmTooLong
        }
    }
}
