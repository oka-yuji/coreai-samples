#if os(macOS)
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import OCRDemoRuntime
    import OCRUI
    import SwiftUI

    @MainActor
    final class OCRMacAdapter: ModelDemoAdapter {
        let demoID = TeleOCRDemoCatalog.selection.demoID
        private let sessions = OCRDemoSessions()

        func preview(for selection: DemoSelection) -> AnyView {
            AnyView(Text("Open TeleOCR to read a document image."))
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            guard gate.session == session, gate.isPermitted(),
                let store = sessions.bind(session, permits: gate.isPermitted)
            else {
                return AnyView(Text("Close and reopen the OCR demo to start this session."))
            }
            return AnyView(OCRPageView(store: store))
        }

        func stop(_ session: ModelDemoSession) async throws { await sessions.stop(session) }
    }
#endif
