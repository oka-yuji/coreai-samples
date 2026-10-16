import ObjectFinderCore
import Testing

struct ObjectPreviewExpiryTests {
    private let candidate = ObjectCandidate(
        id: 1, region: .init(x: 0.1, y: 0.2, width: 0.3, height: 0.4), similarity: 0.9)

    @Test("Only visible, unexpired candidates in the expected generation require an expiry deadline")
    func deadlines() {
        var gate = ObjectPreviewGate()
        #expect(gate.expiration(at: 10) == nil)
        gate.receive([candidate], generation: 1, validUntil: 12)
        #expect(gate.expiration(at: 10) == nil)
        gate.expect(1)
        #expect(gate.expiration(at: 10) == 12)
        #expect(gate.expiration(at: 11) == 12)
        #expect(gate.expiration(at: 12.001) == nil)
        #expect(gate.expiration(at: .nan) == nil)
        gate.receive([], generation: 1, validUntil: 12)
        #expect(gate.expiration(at: 10) == nil)
        gate.receive([candidate], generation: 1, validUntil: .infinity)
        #expect(gate.expiration(at: 10) == nil)
    }

    @Test("Clearing, rotating or advancing generations cannot revive an old expiry or candidate")
    func invalidation() {
        var gate = ObjectPreviewGate()
        gate.expect(1)
        gate.receive([candidate], generation: 1, validUntil: 12)
        gate.expect(nil)
        #expect(gate.expiration(at: 10) == nil)
        gate.expect(3)
        gate.receive([candidate], generation: 1, validUntil: 14)
        #expect(gate.expiration(at: 10) == nil)
        gate.receive([candidate], generation: 3, validUntil: 13)
        #expect(gate.expiration(at: 10) == 13)
        gate.receive([candidate], generation: 1, validUntil: 20)
        #expect(gate.expiration(at: 10) == 13)
        #expect(gate.visible(at: 13.1).isEmpty)
    }
}
