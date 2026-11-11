#if os(iOS)
    import AVFoundation
    import ObjectFinderCore
    import ObjectFinderFeature
    import UIKit

    public final class ObjectFinderPreview: @unchecked Sendable {
        let session: AVCaptureSession
        @MainActor private let layer: AVCaptureVideoPreviewLayer
        @MainActor private let boxes = CAShapeLayer()
        @MainActor private var rotation = 90.0
        @MainActor private var gate = ObjectPreviewGate()
        @MainActor private var minimumSimilarity = 0.67
        @MainActor private var maximumBoxes = 3
        @MainActor private var lastDrawn: [ObjectCandidate] = []
        @MainActor private var lastBounds = CGRect.null
        @MainActor private var lastRotation = Double.nan
        @MainActor private var expiryTimer: Timer?
        @MainActor private var expiryDeadline: Double?
        @MainActor private var activeSession: Int?
        @MainActor private var cameraID: String?
        @MainActor private var receive: ((ObjectFinderAction) -> Void)?
        @MainActor private lazy var rotationObserver = ObjectFinderRotationObserver(
            sourceChanged: { [weak self] in self?.receive?(.rotationSourceChanged($0)) },
            previewChanged: { [weak self] in self?.setPreviewRotation($0) },
            captureChanged: { [weak self] in self?.setCaptureRotation($1, source: $0) })

        @MainActor
        public init() {
            let session = AVCaptureSession()
            self.session = session
            layer = AVCaptureVideoPreviewLayer(session: session)
            layer.videoGravity = .resizeAspect
            boxes.strokeColor = UIColor.systemYellow.cgColor
            boxes.fillColor = UIColor.clear.cgColor
            boxes.lineWidth = 3
            layer.addSublayer(boxes)
        }

        @MainActor
        public func attach(to parent: CALayer, receive: @escaping (ObjectFinderAction) -> Void) {
            if layer.superlayer !== parent {
                rotationObserver.unmount()
                parent.addSublayer(layer)
            }
            self.receive = receive
            configureRotation()
        }

        @MainActor
        public func detach(from parent: CALayer) {
            guard layer.superlayer === parent else { return }
            rotationObserver.unmount()
            receive = nil
            gate.expect(nil)
            redraw()
            layer.removeFromSuperlayer()
        }

        @MainActor
        func beginRotation(session: Int) {
            rotationObserver.end()
            activeSession = session
            cameraID = nil
            gate.prepareCapture()
        }

        @MainActor
        func endRotation() {
            activeSession = nil
            cameraID = nil
            rotationObserver.end()
            gate.expect(nil)
            redraw()
        }

        @MainActor
        func connected(cameraID: String, generation: Int, rotation: Double) throws {
            guard let activeSession else { throw CancellationError() }
            gate.begin(generation, rotation: rotation)
            try rotationObserver.select(cameraID: cameraID, session: activeSession)
            self.cameraID = cameraID
            configureRotation()
            applyRotation()
            redraw()
        }

        @MainActor
        private func configureRotation() {
            guard layer.superlayer != nil, receive != nil, activeSession != nil, cameraID != nil else { return }
            rotationObserver.mount(layer)
        }

        @MainActor
        private func setPreviewRotation(_ angle: Double) {
            rotation = angle
            applyRotation()
            redraw()
        }

        @MainActor
        private func setCaptureRotation(_ angle: Double, source: ObjectCameraRotationSource) {
            guard activeSession == source.session else { return }
            gate.updateCaptureAngle(angle)
            receive?(.captureAngleChanged(source, angle))
            redraw()
        }

        @MainActor
        private func applyRotation() {
            if let connection = layer.connection, connection.isVideoRotationAngleSupported(rotation) {
                if connection.videoRotationAngle != rotation { connection.videoRotationAngle = rotation }
            }
        }

        func publish(_ candidates: [ObjectCandidate], generation: Int, rotation: Double, validUntil: Double = 0) {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.gate.receive(
                    candidates, generation: generation, validUntil: validUntil, rotation: rotation)
                self.redraw()
            }
        }

        @MainActor
        public func layout(
            in bounds: CGRect, minimumSimilarity: Double, maximumBoxes: Int, expectedGeneration: Int?, parent: CALayer
        ) {
            guard layer.superlayer === parent else { return }
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            layer.frame = bounds
            boxes.frame = layer.bounds
            self.minimumSimilarity = minimumSimilarity
            self.maximumBoxes = maximumBoxes
            let generation = activeSession != nil && cameraID != nil ? expectedGeneration : nil
            gate.expect(generation)
            boxes.isHidden = generation == nil
            applyRotation()
            redraw()
            CATransaction.commit()
        }

        @MainActor
        private func redraw() {
            let now = ProcessInfo.processInfo.systemUptime
            let candidates = gate.visible(at: now, rotation: rotation)
            let selected = ObjectCandidateSelection.select(
                candidates, minimumSimilarity: minimumSimilarity, limit: maximumBoxes)
            scheduleExpiry(
                selected.isEmpty || layer.superlayer == nil ? nil : gate.expiration(at: now, rotation: rotation),
                now: now)
            guard selected != lastDrawn || layer.bounds != lastBounds || rotation != lastRotation else { return }
            lastDrawn = selected
            lastBounds = layer.bounds
            lastRotation = rotation
            let path = UIBezierPath()
            for candidate in selected {
                guard let box = ObjectTrackingPolicy.metadataRegion(candidate.region, rotation: rotation) else {
                    continue
                }
                let rect = layer.layerRectConverted(
                    fromMetadataOutputRect:
                        CGRect(x: box.x, y: box.y, width: box.width, height: box.height))
                path.append(UIBezierPath(roundedRect: rect, cornerRadius: 8))
            }
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            boxes.path = path.cgPath
            CATransaction.commit()
        }

        @MainActor
        private func scheduleExpiry(_ deadline: Double?, now: Double) {
            guard deadline != expiryDeadline else { return }
            expiryTimer?.invalidate()
            expiryTimer = nil
            expiryDeadline = deadline
            guard let deadline else { return }
            let timer = Timer(
                timeInterval: max(0.001, deadline - now + 0.001), target: ExpiryTarget(preview: self),
                selector: #selector(ExpiryTarget.fire(_:)), userInfo: nil, repeats: false)
            expiryTimer = timer
            RunLoop.main.add(timer, forMode: .common)
        }

        @MainActor
        private final class ExpiryTarget: NSObject {
            weak var preview: ObjectFinderPreview?

            init(preview: ObjectFinderPreview) { self.preview = preview }

            @objc func fire(_ timer: Timer) {
                guard let preview, preview.expiryTimer === timer else { return }
                preview.expiryTimer = nil
                preview.expiryDeadline = nil
                preview.redraw()
            }
        }
    }
#endif
