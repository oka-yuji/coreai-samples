public struct VoxCPMReferenceConfiguration: Decodable, Equatable, Sendable {
    static let pinned = VoxCPMReferenceConfiguration(
        sampleRate: 16_000, minimumSamples: 80_000, maximumSamples: 160_000, hopSamples: 640, startToken: 103,
        endToken: 104, padding: "right_zero")
    public static let paddedSamples = 161_280

    public let sampleRate: Int
    public let minimumSamples: Int
    public let maximumSamples: Int
    public let hopSamples: Int
    public let startToken: Int32
    public let endToken: Int32
    public let padding: String

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case minimumSamples = "minimum_samples"
        case maximumSamples = "maximum_samples"
        case hopSamples = "hop_samples"
        case startToken = "start_token"
        case endToken = "end_token"
        case padding
    }
}
