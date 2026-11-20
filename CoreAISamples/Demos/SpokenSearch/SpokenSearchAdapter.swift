import ModelLibraryFeature
import ModelLibraryRuntime
import SpokenSearchRuntime
import SwiftUI

final class SpokenSearchAdapter: ModelDemoAdapter {
    let demoID = SpokenSearchCatalog.selection.demoID
    private var sessions: [UUID: (model: ModelDemoSession, search: SpokenSearchSession)] = [:]
    private var retiredThrough: [UUID: Int] = [:]

    func preview(for selection: DemoSelection) -> AnyView {
        guard selection == SpokenSearchCatalog.selection else { return AnyView(EmptyView()) }
        return AnyView(SpokenSearchPreview())
    }

    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        guard let search = bind(session, gate: gate) else {
            return AnyView(Text("Reopen spoken word search and choose the recording again."))
        }
        return AnyView(SpokenSearchScreen(store: search.store).id(session.id))
    }

    func bind(_ session: ModelDemoSession, gate: DemoExecutionGate) -> SpokenSearchSession? {
        let ticket = session.id
        guard session.selection == SpokenSearchCatalog.selection, gate.session == session,
            ticket.number > retiredThrough[ticket.owner, default: 0], gate.isPermitted()
        else { return nil }
        if let entry = sessions[ticket.owner] { return entry.model == session ? entry.search : nil }
        let search = SpokenSearchSession { [weak self] in
            guard let self, sessions[ticket.owner]?.model == session else { return false }
            return ticket.number > retiredThrough[ticket.owner, default: 0] && gate.isPermitted()
        }
        sessions[ticket.owner] = (session, search)
        return search
    }

    func stop(_ session: ModelDemoSession) async {
        let ticket = session.id
        guard let entry = sessions[ticket.owner], entry.model == session else { return }
        retiredThrough[ticket.owner] = ticket.number
        await entry.search.close()
        if sessions[ticket.owner]?.search === entry.search { sessions[ticket.owner] = nil }
    }
}
