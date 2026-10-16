import Foundation

public func resolveFrames(logFrames: Float, fixed: Int?) throws -> Int {
    if let fixed {
        guard (13...750).contains(fixed) else { throw SpeechError.invalidDuration }
        return fixed
    }
    let value = expm1f(logFrames)
    guard value.isFinite else { throw SpeechError.invalidDuration }
    return Int(max(13, min(750, value.rounded(.toNearestOrEven))))
}

public struct RecyclePolicy: Sendable {
    public private(set) var completed = 0
    public private(set) var previousFrames: Int?
    public init() {}
    public func needsReload(for frames: Int) -> Bool {
        completed >= 60 || previousFrames.map { max($0, frames) >= 2 * min($0, frames) } == true
    }
    public mutating func reset() {
        completed = 0
        previousFrames = nil
    }
    public mutating func recordCompletion(frames: Int) {
        completed += 1
        previousFrames = frames
    }
}

public func floatWAV(samples: [Float]) throws -> Data {
    guard !samples.isEmpty, samples.count <= 750 * 1920, samples.allSatisfy({ $0.isFinite }) else {
        throw SpeechError.invalidSamples
    }
    var data = Data()
    func string(_ value: String) { data.append(contentsOf: value.utf8) }
    func integer<T: FixedWidthInteger>(_ value: T) {
        var value = value.littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    let bytes = UInt32(samples.count * 4)
    string("RIFF")
    integer(bytes + 50)
    string("WAVEfmt ")
    integer(UInt32(18))
    integer(UInt16(3))
    integer(UInt16(1))
    integer(UInt32(48000))
    integer(UInt32(192000))
    integer(UInt16(4))
    integer(UInt16(32))
    integer(UInt16(0))
    string("fact")
    integer(UInt32(4))
    integer(UInt32(samples.count))
    string("data")
    integer(bytes)
    for sample in samples { integer(sample.bitPattern) }
    return data
}
