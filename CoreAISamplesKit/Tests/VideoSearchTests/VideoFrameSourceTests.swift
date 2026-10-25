import AVFoundation
import CoreGraphics
import Foundation
import Testing
import VideoSearchFeature
import VideoSearchRuntime

@Suite("Video frame source contracts", .timeLimit(.minutes(1)))
struct VideoFrameSourceTests {
    let maximumSize = CGSize(width: 640, height: 640)

    @Test(
        "VID-05 A synthetic three-second video yields its length and upright frames at the sampled times in red, green and blue order"
    )
    func colorsInOrder() async throws {
        let url = try await SyntheticVideo.colors(width: 160, height: 120)
        defer { SyntheticVideo.remove(url) }
        let duration = try await VideoFrameSource.duration(of: url)
        #expect(abs(duration - 3) <= 0.2)
        let times = VideoSampling.times(duration: duration)
        #expect(times.count == 3)
        var frames: [(time: Double, image: CGImage)] = []
        for try await frame in VideoFrameSource.frames(of: url, at: times, maximumSize: maximumSize) {
            frames.append(frame)
        }
        #expect(frames.map { SyntheticVideo.dominantColor(of: $0.image) } == [.red, .green, .blue])
        #expect(frames.allSatisfy { $0.image.width == 160 && $0.image.height == 120 })
        #expect(zip(frames, times).allSatisfy { abs($0.time - $1) <= 0.5 })
    }

    @Test("VID-05 A video recorded with a quarter-turn transform yields portrait frames in the same color order")
    func rotated() async throws {
        let url = try await SyntheticVideo.colors(
            width: 160, height: 120, transform: CGAffineTransform(rotationAngle: .pi / 2))
        defer { SyntheticVideo.remove(url) }
        let times = VideoSampling.times(duration: try await VideoFrameSource.duration(of: url))
        var frames: [(time: Double, image: CGImage)] = []
        for try await frame in VideoFrameSource.frames(of: url, at: times, maximumSize: maximumSize) {
            frames.append(frame)
        }
        #expect(frames.map { SyntheticVideo.dominantColor(of: $0.image) } == [.red, .green, .blue])
        #expect(frames.allSatisfy { $0.image.width == 120 && $0.image.height == 160 })
    }

    @Test("VID-05 Times requested across several batches yield one frame each in order with the color of its own time")
    func batches() async throws {
        let url = try await SyntheticVideo.colors(width: 160, height: 120)
        defer { SyntheticVideo.remove(url) }
        let times = (0..<10).map { 0.2 + Double($0) * 0.3 }
        var frames: [(time: Double, image: CGImage)] = []
        for try await frame in VideoFrameSource.frames(of: url, at: times, maximumSize: maximumSize) {
            frames.append(frame)
        }
        #expect(frames.count == times.count)
        #expect(zip(frames, frames.dropFirst()).allSatisfy { $0.time <= $1.time })
        for frame in frames {
            let expected: SyntheticVideo.Color = frame.time < 1 ? .red : frame.time < 2 ? .green : .blue
            #expect(SyntheticVideo.dominantColor(of: frame.image) == expected, "\(frame.time)")
        }
    }

    @Test("VID-05 Cancelling the reading task after the first frame stops the frames before the rest are read")
    func cancellation() async throws {
        let url = try await SyntheticVideo.colors(width: 160, height: 120)
        defer { SyntheticVideo.remove(url) }
        let times = VideoSampling.times(duration: try await VideoFrameSource.duration(of: url))
        let first = AsyncStream<Void>.makeStream()
        let hold = AsyncStream<Void>.makeStream()
        let maximumSize = maximumSize
        let reading = Task {
            var count = 0
            do {
                for try await _ in VideoFrameSource.frames(of: url, at: times, maximumSize: maximumSize) {
                    count += 1
                    guard count == 1 else { continue }
                    first.continuation.yield(())
                    var waiting = hold.stream.makeAsyncIterator()
                    _ = await waiting.next()
                }
            } catch is CancellationError {
                return (count, true)
            }
            return (count, false)
        }
        var started = first.stream.makeAsyncIterator()
        _ = await started.next()
        reading.cancel()
        hold.continuation.yield(())
        let (count, cancelled) = try await reading.value
        #expect(times.count == 3 && count == 1 && cancelled)
    }

    @Test("VID-05 An audio-only file is rejected as an invalid video and an unreadable or missing file fails")
    func invalidVideo() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "video-search-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let audio = folder.appending(path: "tone.caf")
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16_000))
        buffer.frameLength = 16_000
        try AVAudioFile(forWriting: audio, settings: format.settings).write(from: buffer)
        await #expect(throws: VideoSearchError.invalidVideo) { try await VideoFrameSource.duration(of: audio) }
        let text = folder.appending(path: "notes.mov")
        try Data("not a movie".utf8).write(to: text)
        await #expect(throws: (any Error).self) { try await VideoFrameSource.duration(of: text) }
        await #expect(throws: (any Error).self) {
            try await VideoFrameSource.duration(of: folder.appending(path: "missing.mov"))
        }
    }
}

enum SyntheticVideo {
    enum Color: Equatable { case red, green, blue, other }

    static func colors(width: Int, height: Int, transform: CGAffineTransform = .identity) async throws -> URL {
        let images = try [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)].map { red, green, blue in
            try solid(CGColor(red: red, green: green, blue: blue, alpha: 1), width: width, height: height)
        }
        return try await write(width: width, height: height, frames: 30, transform: transform) { images[$0 / 10] }
    }

    static func write(
        width: Int, height: Int, frames: Int, transform: CGAffineTransform = .identity, image: (Int) -> CGImage
    ) async throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appending(path: "video-search-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appending(path: "clip.mov")
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
                AVVideoColorPropertiesKey: [
                    AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                    AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                    AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
                ],
            ])
        input.transform = transform
        let receiver = writer.inputPixelBufferReceiver(for: input, pixelBufferAttributes: nil)
        try writer.start()
        writer.startSession(atSourceTime: .zero)
        for frame in 0..<frames {
            let buffer = try pixelBuffer(width: width, height: height, image: image(frame))
            try await receiver.append(
                CVReadOnlyPixelBuffer(unsafeBuffer: buffer), with: CMTime(value: CMTimeValue(frame), timescale: 10))
        }
        receiver.finish()
        writer.endSession(atSourceTime: CMTime(value: CMTimeValue(frames), timescale: 10))
        await writer.finishWriting()
        guard writer.status == .completed else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        return url
    }

    static func remove(_ url: URL) {
        try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
    }

    static func dominantColor(of image: CGImage) -> Color {
        guard let pixel = rgb(of: image) else { return .other }
        let (red, green, blue) = (pixel[0], pixel[1], pixel[2])
        if red > 160 && green < 96 && blue < 96 { return .red }
        if green > 160 && red < 96 && blue < 96 { return .green }
        if blue > 160 && red < 96 && green < 96 { return .blue }
        return .other
    }

    static func rgb(of image: CGImage, x: Double = 0.5, y: Double = 0.5) -> [Int]? {
        var pixel = [UInt8](repeating: 0, count: 4)
        let drawn = pixel.withUnsafeMutableBytes { bytes -> Bool in
            guard
                let context = CGContext(
                    data: bytes.baseAddress, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            let (width, height) = (CGFloat(image.width), CGFloat(image.height))
            context.draw(
                image,
                in: CGRect(x: -(width * x).rounded(), y: -(height * (1 - y)).rounded(), width: width, height: height))
            return true
        }
        return drawn ? pixel.prefix(3).map(Int.init) : nil
    }

    static func solid(_ color: CGColor, width: Int, height: Int) throws -> CGImage {
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw CocoaError(.fileWriteUnknown) }
        context.setFillColor(color)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = context.makeImage() else { throw CocoaError(.fileWriteUnknown) }
        return image
    }

    private static func pixelBuffer(width: Int, height: Int, image: CGImage) throws -> CVPixelBuffer {
        var created: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [String: Any]()] as CFDictionary
        guard
            CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes, &created)
                == kCVReturnSuccess, let buffer = created
        else { throw CocoaError(.fileWriteUnknown) }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { throw CocoaError(.fileWriteUnknown) }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}
