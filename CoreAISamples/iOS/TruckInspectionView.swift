#if os(iOS)
    import FeatureStore
    import SwiftUI
    import TruckInspectionFeature

    struct TruckInspectionView: View {
        let store: Store<TruckInspectionFeature>
        let assetFolders: [URL]
        @Environment(\.scenePhase) private var scenePhase
        @Environment(\.dismiss) private var dismiss

        var body: some View {
            NavigationStack {
                Form {
                    TruckInspectionContent(store: store) {
                        if assetFolders.isEmpty {
                            Text("Place the GLiNER asset folder in this app's Documents, then reopen the app.")
                                .font(.caption)
                        }
                        ForEach(assetFolders, id: \.self) { folder in
                            Button(folder.lastPathComponent) { store.send(.modelChosen(folder)) }
                                .disabled(!store.state.canConfigure)
                        }
                    }
                }
                .navigationTitle("GLiNER2.5-multi")
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
                .task { store.send(.appeared) }
                .onDisappear { store.send(.closed) }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .background { store.send(.stop) }
                }
            }
        }
    }
#endif
