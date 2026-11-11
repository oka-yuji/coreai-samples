#if os(iOS)
    import AVFoundation
    import Foundation
    import ObjectFinderCore
    import ObjectFinderFeature

    @MainActor
    final class ObjectFinderRotationObserver {
        private let relay: ObjectCameraRotationRelay
        private var device: AVCaptureDevice?
        private var session: Int?
        private weak var layer: AVCaptureVideoPreviewLayer?
        private var coordinator: AVCaptureDevice.RotationCoordinator?
        private var observations: [NSKeyValueObservation] = []
        private var generation = 0

        init(
            sourceChanged: @escaping (ObjectCameraRotationSource) -> Void,
            previewChanged: @escaping (Double) -> Void,
            captureChanged: @escaping (ObjectCameraRotationSource, Double) -> Void
        ) {
            relay = ObjectCameraRotationRelay(
                sourceChanged: sourceChanged, previewChanged: previewChanged, captureChanged: captureChanged)
        }

        func select(cameraID: String, session: Int) throws {
            if device?.uniqueID == cameraID, self.session == session { return }
            end()
            guard let device = AVCaptureDevice(uniqueID: cameraID) else { throw ObjectFinderFailure.cameraUnavailable }
            self.device = device
            self.session = session
        }

        func mount(_ layer: AVCaptureVideoPreviewLayer) {
            guard let device, let session else { return }
            if self.layer === layer, coordinator != nil {
                relay.reapplyPreview()
                return
            }
            unmount()
            self.layer = layer
            generation += 1
            let source = ObjectCameraRotationSource(session: session, cameraID: device.uniqueID, generation: generation)
            let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: layer)
            self.coordinator = coordinator
            relay.begin(
                source, preview: Double(coordinator.videoRotationAngleForHorizonLevelPreview),
                capture: Double(coordinator.videoRotationAngleForHorizonLevelCapture))
            guard self.coordinator === coordinator else { return }
            observe(
                \.videoRotationAngleForHorizonLevelPreview, target: .preview, coordinator: coordinator, source: source)
            observe(
                \.videoRotationAngleForHorizonLevelCapture, target: .capture, coordinator: coordinator, source: source)
        }

        func end() {
            unmount()
            device = nil
            session = nil
        }

        func unmount() {
            relay.end()
            for observation in observations { observation.invalidate() }
            observations = []
            coordinator = nil
            layer = nil
        }

        private func observe(
            _ keyPath: KeyPath<AVCaptureDevice.RotationCoordinator, CGFloat>,
            target: ObjectCameraRotationRelay.Target,
            coordinator: AVCaptureDevice.RotationCoordinator, source: ObjectCameraRotationSource
        ) {
            observations.append(
                coordinator.observe(keyPath, options: .new) { [weak self] _, change in
                    guard let angle = change.newValue else { return }
                    MainActor.assumeIsolated { self?.relay.receive(Double(angle), target: target, from: source) }
                })
        }
    }
#endif
