#if os(macOS)
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ReceiptDemoRuntime
    import ReceiptUI
    import SwiftUI

    @MainActor
    final class ReceiptMacAdapter: ModelDemoAdapter {
        let demoID = ReceiptDemoCatalog.selection.demoID
        private let sessions = ReceiptDemoSessions()

        func preview(for selection: DemoSelection) -> AnyView {
            AnyView(Text("Open Receipt Desk to organize receipts on this Mac."))
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            guard gate.session == session, gate.isPermitted(),
                let store = sessions.bind(session, permits: gate.isPermitted)
            else {
                return AnyView(Text("Close and reopen Receipt Desk to start this session."))
            }
            return AnyView(ReceiptPageView(store: store))
        }

        func stop(_ session: ModelDemoSession) async throws { await sessions.stop(session) }
    }
#endif
