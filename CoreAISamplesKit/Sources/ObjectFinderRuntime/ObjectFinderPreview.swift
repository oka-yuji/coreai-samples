#if os(iOS)
    import AVFoundation
    import ObjectFinderCore
    import UIKit

    public final class ObjectFinderPreview: @unchecked Sendable {
        let session: AVCaptureSession
        @MainActor private let layer: AVCaptureVideoPreviewLayer
        @MainActor private let boxes = CAShapeLayer()
        @MainActor private var rotation = 90.0
        @MainActor private var gate = ObjectPreviewGate()
        @MainActor private var minimumSimilarity = 0.67
        @MainActor private var lastDrawn: [ObjectCandidate] = []
        @MainActor private var lastBounds = CGRect.null
        @MainActor private var lastRotation = Double.nan
        @MainActor private var expiryTimer: Timer?
        @MainActor private var expiryDeadline: Double?

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
        public func attach(to parent: CALayer) { parent.addSublayer(layer) }

        @MainActor
        public func detach(from parent: CALayer) {
            guard layer.superlayer === parent else { return }
            gate.expect(nil)
            redraw()
            layer.removeFromSuperlayer()
        }

        @MainActor
        func connected() {
            gate.expect(nil)
            applyRotation()
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
                    rotation == self.rotation ? candidates : [], generation: generation, validUntil: validUntil)
                self.redraw()
            }
        }

        @MainActor
        public func layout(in bounds: CGRect, minimumSimilarity: Double, expectedGeneration: Int?, rotation: Double) {
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            layer.frame = bounds
            boxes.frame = layer.bounds
            self.minimumSimilarity = minimumSimilarity
            self.rotation = rotation
            gate.expect(expectedGeneration)
            boxes.isHidden = expectedGeneration == nil
            applyRotation()
            redraw()
            CATransaction.commit()
        }

        @MainActor
        private func redraw() {
            let now = ProcessInfo.processInfo.systemUptime
            let candidates = gate.visible(at: now)
            let selected = ObjectCandidateSelection.select(candidates, minimumSimilarity: minimumSimilarity)
            scheduleExpiry(selected.isEmpty || layer.superlayer == nil ? nil : gate.expiration(at: now), now: now)
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
