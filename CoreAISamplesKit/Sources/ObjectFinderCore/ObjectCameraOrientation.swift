public enum ObjectCameraPose: Sendable {
    case portrait, landscapeLeft, landscapeRight, portraitUpsideDown, flat
}

public struct ObjectCameraOrientation: Equatable, Sendable {
    public private(set) var angle: Double = 90

    public init() {}

    public mutating func update(_ pose: ObjectCameraPose) {
        switch pose {
        case .portrait: angle = 90
        case .landscapeLeft: angle = 0
        case .landscapeRight: angle = 180
        case .portraitUpsideDown: angle = 270
        case .flat: break
        }
    }
}

public struct ObjectCameraRotation: Equatable, Sendable {
    public let session: Int
    public let revision: Int
    public let angle: Double

    public init(session: Int, revision: Int, angle: Double) {
        self.session = session
        self.revision = revision
        self.angle = angle
    }
}

public struct ObjectCameraConfiguration: Equatable, Sendable {
    public private(set) var rotation = ObjectCameraRotation(session: 0, revision: 0, angle: 90)

    public init() {}

    public mutating func update(_ value: ObjectCameraRotation) {
        guard [0, 90, 180, 270].contains(value.angle), value.session >= 0, value.revision >= 0,
            value.session > rotation.session
                || (value.session == rotation.session && value.revision >= rotation.revision)
        else { return }
        rotation = value
    }
}
