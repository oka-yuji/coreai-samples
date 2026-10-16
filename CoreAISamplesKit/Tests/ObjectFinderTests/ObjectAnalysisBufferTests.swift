import ObjectFinderCore
import Testing

struct ObjectAnalysisBufferTests {
    @Test("Each image request accepts only one eligible latest frame, and capture restart discards pending demand")
    func demandAndRestart() {
        var buffer = ObjectAnalysisBuffer<Int>()
        #expect(buffer.offer(1, at: 0) == nil)
        #expect(buffer.offer(2, at: 0.1) == nil)
        #expect(buffer.request() == 2)
        #expect(buffer.offer(3, at: 0.2) == nil)
        #expect(buffer.offer(4, at: 0.4) == nil)
        #expect(buffer.request() == 4)
        #expect(buffer.request() == nil)
        #expect(buffer.offer(5, at: 0.41) == nil)
        #expect(buffer.offer(6, at: 0.56) == 6)
        #expect(buffer.offer(7, at: 0.8) == nil)
        #expect(buffer.request() == 7)
        #expect(buffer.request() == nil)
        buffer.reset()
        #expect(buffer.latest == nil)
        #expect(buffer.offer(8, at: 0.9) == nil)
        #expect(buffer.request() == 8)
    }

    @Test("A pending image request ignores invalid timestamps and recovers when the camera timestamp restarts")
    func timeBoundaries() {
        var buffer = ObjectAnalysisBuffer<Int>()
        #expect(buffer.request() == nil)
        #expect(buffer.offer(1, at: .nan) == nil)
        #expect(buffer.offer(2, at: .infinity) == nil)
        #expect(buffer.latest == nil)
        #expect(buffer.offer(3, at: 10) == 3)
        #expect(buffer.request() == nil)
        #expect(buffer.offer(4, at: 0) == 4)
        buffer.reset()
        #expect(buffer.request() == nil)
        #expect(buffer.offer(5, at: 0) == 5)
    }
}
