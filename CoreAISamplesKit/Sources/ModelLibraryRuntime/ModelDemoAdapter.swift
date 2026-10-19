import ModelLibraryFeature
import SwiftUI

@MainActor
public protocol ModelDemoAdapter: AnyObject {
    var demoID: String { get }
    func preview(for selection: DemoSelection) -> AnyView
    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView
    func stop(_ session: ModelDemoSession) async throws
}

nonisolated public struct DemoExecutionGate: Sendable {
    public let session: ModelDemoSession
    public let isPermitted: @MainActor @Sendable () -> Bool

    public init(session: ModelDemoSession, isPermitted: @escaping @MainActor @Sendable () -> Bool) {
        self.session = session
        self.isPermitted = isPermitted
    }
}

public final class ModelDemoAdapterRegistry {
    private var adapters: [String: any ModelDemoAdapter] = [:]

    public init() {}

    public var connectedDemos: Set<String> { Set(adapters.keys) }

    public func register(_ adapter: any ModelDemoAdapter) {
        adapters[adapter.demoID] = adapter
    }

    public func adapter(for demoID: String) -> (any ModelDemoAdapter)? { adapters[demoID] }
}
