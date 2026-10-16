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
        public func expire() {
            redraw()
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
            let candidates = gate.visible(at: ProcessInfo.processInfo.systemUptime)
            let selected = ObjectCandidateSelection.select(candidates, minimumSimilarity: minimumSimilarity)
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
    }
#endif
