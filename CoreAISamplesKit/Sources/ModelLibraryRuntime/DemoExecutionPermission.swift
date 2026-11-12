import Observation

@MainActor
public final class DemoExecutionPermission {
    public let isPermitted: @MainActor () -> Bool

    public init(isPermitted: @escaping @MainActor () -> Bool) {
        self.isPermitted = isPermitted
    }
}

@Observable
final class ModelLibraryGateChanges {
    private(set) var generation: UInt64 = 0

    func invalidate() { generation &+= 1 }
}
