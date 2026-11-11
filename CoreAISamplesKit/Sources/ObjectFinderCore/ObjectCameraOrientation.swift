public struct ObjectCameraRotationSource: Equatable, Sendable {
    public let session: Int
    public let cameraID: String
    public let generation: Int

    public init(session: Int, cameraID: String, generation: Int) {
        self.session = session
        self.cameraID = cameraID
        self.generation = generation
    }
}

public struct ObjectCameraOrientation: Equatable, Sendable {
    public private(set) var angle: Double = 90
    public private(set) var source: ObjectCameraRotationSource?

    public init() {}

    public mutating func bind(_ source: ObjectCameraRotationSource?) { self.source = source }

    public mutating func update(_ value: Double, from source: ObjectCameraRotationSource) -> Bool {
        guard self.source == source, let value = Self.normalized(value), value != angle else { return false }
        angle = value
        return true
    }

    public static func normalized(_ value: Double) -> Double? {
        guard value.isFinite else { return nil }
        let remainder = value.truncatingRemainder(dividingBy: 360)
        let positive = remainder < 0 ? remainder + 360 : remainder
        let angle = positive.rounded(.toNearestOrAwayFromZero).truncatingRemainder(dividingBy: 360)
        guard [0, 90, 180, 270].contains(angle) else { return nil }
        return angle == 0 ? 0 : angle
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
