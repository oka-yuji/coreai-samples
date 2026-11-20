#if os(macOS)
    import ModelLibraryFeature
    import SwiftUI

    struct SpokenSearchHost: View {
        let library: MacModelLibrary

        var body: some View {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(LocalizedStringKey(SpokenSearchCatalog.title)).font(.title2.bold())
                    if let store = library.store, let rendering = library.rendering {
                        if store.state.pendingDestination != nil {
                            ProgressView("Finishing the previous demo…")
                            if store.state.failure != nil {
                                Button("Try again") { store.send(.retryDemoStop) }
                            }
                        } else if let session = store.state.session, session.selection == SpokenSearchCatalog.selection
                        {
                            rendering.content(for: session)
                        } else {
                            rendering.preview(for: SpokenSearchCatalog.selection)
                        }
                    } else {
                        SpokenSearchPreview()
                    }
                }
                .frame(maxWidth: 760, alignment: .leading)
                .frame(maxWidth: .infinity)
                .padding(24)
            }
            .demoNavigationTitle("EmbeddingGemma 2")
            .onAppear { library.refresh() }
        }
    }
#endif
