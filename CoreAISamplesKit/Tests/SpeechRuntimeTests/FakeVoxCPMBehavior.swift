@testable import SpeechRuntime

struct FakeVoxCPMBehavior: Sendable {
    var stopPatch: Int? = 3
    var cancelledCall: Int?
    var nonFinite: VoxCPMFunction?
    var nonFiniteOccurrence = 0
}
