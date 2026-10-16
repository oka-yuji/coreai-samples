import Foundation
import SpeechCore
import Testing

@Test("The frame policy handles periodic and shape changes")
func framePolicy() throws {
    #expect(try resolveFrames(logFrames: 4.924296855926514, fixed: nil) == 137)
    #expect(try resolveFrames(logFrames: 0, fixed: nil) == 13)
    #expect(try resolveFrames(logFrames: 10, fixed: nil) == 750)
    #expect(throws: SpeechError.self) { try resolveFrames(logFrames: .nan, fixed: nil) }
    var policy = RecyclePolicy()
    for _ in 0..<59 { policy.recordCompletion(frames: 100) }
    #expect(!policy.needsReload(for: 199))
    #expect(policy.needsReload(for: 200))
    #expect(policy.needsReload(for: 50))
    policy.recordCompletion(frames: 100)
    #expect(policy.needsReload(for: 100))
    policy.reset()
    #expect(!policy.needsReload(for: 750))
}

@Test("Float WAV preserves the sample bits and format fields")
func waveEncoding() throws {
    let data = try floatWAV(samples: [-1, 0, 1])
    func word(_ offset: Int) -> UInt32 {
        data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
    }
    #expect(data.count == 70)
    #expect(word(4) == 62)
    #expect(word(24) == 48000)
    #expect(word(46) == 3)
    #expect(word(54) == 12)
    #expect(word(58) == Float(-1).bitPattern)
    #expect(word(66) == Float(1).bitPattern)
    #expect(throws: SpeechError.self) { try floatWAV(samples: []) }
    #expect(throws: SpeechError.self) { try floatWAV(samples: [.infinity]) }
}
