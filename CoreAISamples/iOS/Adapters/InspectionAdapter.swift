#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import SwiftUI
    import TruckInspectionFeature
    import TruckInspectionRuntime

    @MainActor
    final class InspectionAdapter: ModelDemoAdapter {
        private struct Entry {
            let session: ModelDemoSession
            let store: Store<TruckInspectionFeature>
            let runner: TruckInspectionRunner
        }

        let demoID = "inspection"
        private var entries: [DemoSelection: Entry] = [:]

        func preview(for selection: DemoSelection) -> AnyView {
            AnyView(InspectionPreview(state: entries[selection]?.store.state))
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            let entry: Entry
            if let current = entries[session.selection], current.session.id == session.id {
                entry = current
            } else {
                do {
                    let made = try TruckInspectionAssembly.makeManagedStore(
                        resource: session.resource, gate: gate, retaining: entries[session.selection]?.store.state)
                    entry = Entry(session: session, store: made.store, runner: made.runner)
                    entries[session.selection] = entry
                } catch {
                    return AnyView(
                        ContentUnavailableView(
                            "Could not load the inspection questions", systemImage: "exclamationmark.triangle",
                            description: Text(verbatim: String(describing: error))))
                }
            }
            let store = entry.store
            return AnyView(
                VStack(alignment: .leading, spacing: 12) {
                    TruckInspectionContent(store: store) { EmptyView() }
                }
                .textFieldStyle(.roundedBorder)
                .onAppear { store.send(.appeared) })
        }

        func stop(_ session: ModelDemoSession) async throws {
            guard let entry = entries[session.selection], entry.session.id == session.id else { return }
            entry.store.send(.closed)
            try await entry.runner.release()
        }
    }

    private struct InspectionPreview: View {
        let state: InspectionState?

        var body: some View {
            VStack(alignment: .leading, spacing: 12) {
                Label("Daily inspection results", systemImage: "truck.box.fill").font(.headline)
                if let state {
                    Text("\(state.records.count) of \(state.items.count) recorded")
                        .font(.subheadline).monospacedDigit()
                    ForEach(state.orderedRecords, id: \.itemID) { record in
                        HStack(alignment: .top) {
                            Text(verbatim: record.title)
                            Spacer()
                            Text(verbatim: state.catalog.title(for: record.candidate.status)).fontWeight(.semibold)
                        }
                    }
                }
                TextField("Or type what you found", text: .constant("")).textFieldStyle(.roundedBorder)
                Button("Start inspection") {}.buttonStyle(.borderedProminent)
            }
        }
    }
#endif
