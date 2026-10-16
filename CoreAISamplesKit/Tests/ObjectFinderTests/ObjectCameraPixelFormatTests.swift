import ObjectFinderCore
import Testing

struct ObjectCameraPixelFormatTests {
    @Test(
        "A supported native 8-bit YUV format is selected, while unknown or unsupported native formats fall back to BGRA"
    )
    func nativeAndFallback() {
        let all: Set<ObjectCameraPixelFormat> = [.bgra, .yuvVideoRange, .yuvFullRange]
        #expect(ObjectCameraPixelFormat.select(native: .yuvVideoRange, supported: all) == .yuvVideoRange)
        #expect(ObjectCameraPixelFormat.select(native: .yuvFullRange, supported: all) == .yuvFullRange)
        #expect(ObjectCameraPixelFormat.select(native: .bgra, supported: all) == .bgra)
        #expect(ObjectCameraPixelFormat.select(native: nil, supported: all) == .bgra)
        #expect(
            ObjectCameraPixelFormat.select(native: .yuvFullRange, supported: [.bgra, .yuvVideoRange]) == .bgra)
        #expect(ObjectCameraPixelFormat.select(native: .yuvVideoRange, supported: [.bgra]) == .bgra)
    }

    @Test("An unavailable compatible output format is rejected instead of silently choosing an unverified format")
    func unsupported() {
        #expect(ObjectCameraPixelFormat.select(native: nil, supported: [.yuvVideoRange]) == nil)
        #expect(ObjectCameraPixelFormat.select(native: .yuvFullRange, supported: [.yuvVideoRange]) == nil)
        #expect(ObjectCameraPixelFormat.select(native: .bgra, supported: []) == nil)
        #expect(ObjectCameraPixelFormat.select(native: .yuvVideoRange, supported: [.yuvVideoRange]) == .yuvVideoRange)
    }
}
