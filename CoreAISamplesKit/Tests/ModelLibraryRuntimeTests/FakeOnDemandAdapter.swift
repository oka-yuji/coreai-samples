import ModelLibraryFeature
import ModelLibraryRuntime
import SwiftUI

@MainActor
final class FakeOnDemandAdapter: ModelOnDemandDemoAdapter {
    let demoID = "comparison"
    var access: OnDemandModelAccess?
    var ordinaryPreviews = 0

    func preview(for selection: DemoSelection, access: OnDemandModelAccess?) -> AnyView {
        self.access = access
        return AnyView(EmptyView())
    }

    func preview(for selection: DemoSelection) -> AnyView {
        ordinaryPreviews += 1
        return AnyView(EmptyView())
    }

    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView { AnyView(EmptyView()) }
    func stop(_ session: ModelDemoSession) async throws {}
}
