import ObjectFinderCore

@MainActor
final class ObjectCameraRotationRelay {
    enum Target: Sendable { case preview, capture }

    private var source: ObjectCameraRotationSource?
    private var previewAngle: Double?
    private var captureAngle: Double?
    private let sourceChanged: (ObjectCameraRotationSource) -> Void
    private let previewChanged: (Double) -> Void
    private let captureChanged: (ObjectCameraRotationSource, Double) -> Void

    init(
        sourceChanged: @escaping (ObjectCameraRotationSource) -> Void,
        previewChanged: @escaping (Double) -> Void,
        captureChanged: @escaping (ObjectCameraRotationSource, Double) -> Void
    ) {
        self.sourceChanged = sourceChanged
        self.previewChanged = previewChanged
        self.captureChanged = captureChanged
    }

    func begin(_ source: ObjectCameraRotationSource, preview: Double, capture: Double) {
        self.source = source
        previewAngle = nil
        captureAngle = nil
        sourceChanged(source)
        receive(preview, target: .preview, from: source)
        receive(capture, target: .capture, from: source)
    }

    func receive(_ value: Double, target: Target, from source: ObjectCameraRotationSource) {
        guard self.source == source, let angle = ObjectCameraOrientation.normalized(value) else { return }
        switch target {
        case .preview:
            guard angle != previewAngle else { return }
            previewAngle = angle
            previewChanged(angle)
        case .capture:
            guard angle != captureAngle else { return }
            captureAngle = angle
            captureChanged(source, angle)
        }
    }

    func reapplyPreview() {
        if source != nil, let previewAngle { previewChanged(previewAngle) }
    }

    func end() {
        source = nil
        previewAngle = nil
        captureAngle = nil
    }
}
