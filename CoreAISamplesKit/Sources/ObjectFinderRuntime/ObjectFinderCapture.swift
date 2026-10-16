#if os(iOS)
    import AVFoundation
    import CoreImage
    import Foundation
    import ObjectFinderCore
    import ObjectFinderFeature

    actor ObjectFinderCapture {
        private struct CameraImage: Sendable {
            let image: CIImage
            let capturedAt: Double
        }

        nonisolated private let queue = DispatchSerialQueue(label: "ObjectFinderCapture")
        nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

        private let session: AVCaptureSession
        private let preview: ObjectFinderPreview
        private let context = CIContext()
        private let output = AVCaptureVideoDataOutput()
        private var delegate: ObjectFinderCaptureDelegate?
        private var continuation: AsyncThrowingStream<CapturedObjectFrame, any Error>.Continuation?
        private var observers: [any NSObjectProtocol] = []
        private var frames = ObjectAnalysisBuffer<CameraImage>()
        private var generation = 0
        private var rotation = 90.0
        private var failure: ObjectFinderFailure?
        private var tracker: ObjectFinderTracker?

        init(preview: ObjectFinderPreview) {
            self.preview = preview
            session = preview.session
        }

        func start(rotation: Double) throws -> AsyncThrowingStream<CapturedObjectFrame, any Error> {
            guard continuation == nil else { throw ObjectFinderError.busy }
            generation += 1
            let generation = generation
            let channel = AsyncThrowingStream<CapturedObjectFrame, any Error>.makeStream(
                bufferingPolicy: .bufferingNewest(1))
            continuation = channel.continuation
            frames.reset()
            failure = nil
            do {
                let device = try configure(rotation: rotation, generation: generation)
                try Self.configureFrameRate(device)
                for name in [AVCaptureSession.wasInterruptedNotification, AVCaptureSession.runtimeErrorNotification] {
                    let token = NotificationCenter.default.addObserver(forName: name, object: session, queue: nil) {
                        [weak self] _ in
                        guard let self else { return }
                        self.queue.async {
                            self.assumeIsolated { capture in
                                guard capture.generation == generation else { return }
                                capture.fail(.cameraInterrupted)
                            }
                        }
                    }
                    observers.append(token)
                }
                session.startRunning()
                guard session.isRunning else { throw ObjectFinderFailure.cameraUnavailable }
                return channel.stream
            } catch {
                stop()
                throw error
            }
        }

        private func setRotation(_ angle: Double) {
            guard let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(angle)
            else { return }
            if connection.videoRotationAngle != angle { connection.videoRotationAngle = angle }
            rotation = angle
        }

        func checkActive() throws {
            if let failure { throw failure }
            guard continuation != nil else { throw CancellationError() }
        }

        func requestFrame() throws {
            try checkActive()
            if let frame = frames.request() { deliver(frame) }
        }

        func stop() {
            generation += 1
            tracker = nil
            frames.reset()
            preview.publish([], generation: generation, rotation: rotation)
            continuation?.finish()
            continuation = nil
            output.setSampleBufferDelegate(nil, queue: nil)
            session.stopRunning()
            for observer in observers { NotificationCenter.default.removeObserver(observer) }
            observers = []
            session.beginConfiguration()
            for input in session.inputs { session.removeInput(input) }
            for item in session.outputs { session.removeOutput(item) }
            session.commitConfiguration()
            delegate = nil
        }

        private func configure(rotation: Double, generation: Int) throws -> AVCaptureDevice {
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                throw ObjectFinderFailure.cameraUnavailable
            }
            let input = try AVCaptureDeviceInput(device: device)
            session.beginConfiguration()
            defer { session.commitConfiguration() }
            session.sessionPreset = .hd1280x720
            guard session.canAddInput(input), session.canAddOutput(output) else {
                throw ObjectFinderFailure.cameraUnavailable
            }
            session.addInput(input)
            session.addOutput(output)
            output.alwaysDiscardsLateVideoFrames = true
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            let delegate = ObjectFinderCaptureDelegate { [weak self] image, time in
                self?.assumeIsolated { capture in
                    guard capture.generation == generation else { return }
                    capture.receive(image, time: time)
                }
            }
            self.delegate = delegate
            output.setSampleBufferDelegate(delegate, queue: queue)
            setRotation(rotation)
            return device
        }

        private static func configureFrameRate(_ device: AVCaptureDevice) throws {
            guard
                device.activeFormat.videoSupportedFrameRateRanges.contains(where: {
                    $0.minFrameRate <= 30 && $0.maxFrameRate >= 30
                })
            else {
                throw ObjectFinderFailure.operation("The selected camera format does not support 30 fps")
            }
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            if device.activeFormat.isAutoVideoFrameRateSupported { device.isAutoVideoFrameRateEnabled = false }
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        }

        func trackCandidates(_ candidates: [ObjectCandidate], frame: CapturedObjectFrame) -> [ObjectCandidate] {
            guard continuation != nil, failure == nil, frame.rotation == rotation,
                ObjectTrackingPolicy.accepts(
                    capturedAt: frame.capturedAt, now: ProcessInfo.processInfo.systemUptime,
                    generation: frame.generation, currentGeneration: generation)
            else { return [] }
            do {
                tracker = nil
                guard !candidates.isEmpty else {
                    preview.publish([], generation: generation, rotation: rotation)
                    return []
                }
                tracker = try ObjectFinderTracker(candidates: candidates, frame: frame)
                return updateTracking(frames.latest?.image ?? CIImage(cgImage: frame.image))
            } catch {
                tracker = nil
                preview.publish([], generation: generation, rotation: rotation)
                return []
            }
        }

        private func updateTracking(_ image: CIImage) -> [ObjectCandidate] {
            do {
                let candidates = try tracker?.track(image, now: ProcessInfo.processInfo.systemUptime) ?? []
                let validUntil = (tracker?.capturedAt ?? 0) + ObjectTrackingPolicy.maximumAge
                if candidates.isEmpty { tracker = nil }
                preview.publish(
                    candidates, generation: generation, rotation: rotation,
                    validUntil: validUntil)
                return candidates
            } catch {
                tracker = nil
                preview.publish([], generation: generation, rotation: rotation)
                return []
            }
        }

        private func receive(_ input: CIImage, time: Double) {
            guard continuation != nil, failure == nil else { return }
            let capturedAt = ProcessInfo.processInfo.systemUptime
            let portrait = rotation == 90 || rotation == 270
            guard portrait == (input.extent.height > input.extent.width) else { return }
            let requested = frames.offer(CameraImage(image: input, capturedAt: capturedAt), at: time)
            if tracker != nil { _ = updateTracking(input) }
            if let requested { deliver(requested) }
        }

        private func deliver(_ frame: CameraImage) {
            guard let image = context.createCGImage(frame.image, from: frame.image.extent) else {
                fail(.operation("Could not read the camera image"))
                return
            }
            continuation?.yield(
                CapturedObjectFrame(
                    image: image, rotation: rotation, generation: generation,
                    capturedAt: frame.capturedAt))
        }

        private func fail(_ error: ObjectFinderFailure) {
            failure = error
            frames.reset()
            tracker = nil
            preview.publish([], generation: generation, rotation: rotation)
            continuation?.finish(throwing: error)
        }
    }

    private final class ObjectFinderCaptureDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, Sendable {
        let receive: @Sendable (CIImage, Double) -> Void
        init(receive: @escaping @Sendable (CIImage, Double) -> Void) { self.receive = receive }
        func captureOutput(
            _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection
        ) {
            guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            receive(CIImage(cvPixelBuffer: pixels), CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds)
        }
    }
#endif
