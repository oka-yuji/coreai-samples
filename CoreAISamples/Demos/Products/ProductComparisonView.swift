import FeatureStore
import Foundation
import ProductCore
import ProductFeature
import SwiftUI
import UniformTypeIdentifiers

struct ProductComparisonView: View {
    let store: Store<ProductFeature>
    #if os(iOS)
        var assetFolders: [URL] = []
        @State private var importingModel = false
    #else
        @Environment(MacModelLibrary.self) private var library: MacModelLibrary?
        @State private var attempted: URL?
    #endif
    @State private var addingProduct = false

    private var state: ProductState { store.state }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header
                ProductStatus(store: store) { addingProduct = true }
                ProductComparisonCards(store: store)
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(Color.primary.opacity(0.025))
        #if os(iOS)
            .onAppear { store.send(.appeared) }
            .onDisappear { store.send(.closed) }
            .fileImporter(isPresented: $importingModel, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { store.send(.modelChosen(url)) }
            }
        #else
            .onAppear {
                store.send(.appeared)
                library?.use(.gliner)
            }
            .onDisappear {
                store.send(.closed)
                library?.stopUsing(.gliner)
                attempted = nil
            }
            .onChange(of: library?.resource(inUse: .gliner)?.directory) { _, new in
                if new == nil { attempted = nil }
            }
            .onChange(of: modelToChoose, initial: true) { _, url in
                guard let url else { return }
                attempted = url
                store.send(.modelChosen(url))
            }
        #endif
        .sheet(isPresented: $addingProduct) {
            NavigationStack {
                ScrollView {
                    ProductEntry(store: store) { addingProduct = true }.padding(20)
                }
                .demoNavigationTitle("Add a product")
                .toolbar { Button("Done") { addingProduct = false } }
            }
            .frame(idealWidth: 660, idealHeight: 430)
        }
        .onChange(of: state.cards.count) { oldCount, newCount in
            if newCount > oldCount { addingProduct = false }
        }
        .demoNavigationTitle("GLiNER2.5-multi")
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("GLiNER2.5-multi").font(.title2.bold())
                    Text("Compare product descriptions").font(.headline)
                    Text("Bring the specs together. Keep the source within reach.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Spacer(minLength: 12)
                VStack(alignment: .trailing, spacing: 8) {
                    Button("Add a product", systemImage: "plus") { addingProduct = true }
                        .buttonStyle(.borderedProminent)
                        .disabled(
                            state.preview || state.busy || state.cards.count >= ProductState.cardLimit
                                || state.modelFolder == nil || !canUseModel)
                    Text("\(state.cards.count) / \(ProductState.cardLimit) cards")
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
            if state.preview {
                Label("Preview data — no model has run", systemImage: "eye")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                HStack(spacing: 10) {
                    Image(systemName: "cube.transparent")
                    if let guidance {
                        guidance
                    } else if let folder = state.modelFolder {
                        Text(verbatim: modelName(folder)).lineLimit(1).truncationMode(.middle)
                    }
                    Spacer()
                    #if os(iOS)
                        if !assetFolders.isEmpty {
                            Menu("Model folder") {
                                ForEach(assetFolders, id: \.self) { url in
                                    Button(url.lastPathComponent) { store.send(.modelChosen(url)) }
                                }
                            }
                        } else {
                            Button("Model folder", systemImage: "folder") { importingModel = true }
                        }
                    #endif
                }
                .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var guidance: Text? {
        #if os(iOS)
            return state.modelFolder == nil ? Text("Choose a GLiNER model folder to begin") : nil
        #else
            guard let library, library.installed(.gliner) == nil else { return nil }
            return Text("Download \(DemoModel.gliner.title) from its heading in the sidebar.")
        #endif
    }

    private var canUseModel: Bool {
        #if os(iOS)
            return true
        #else
            guard let library else { return true }
            return library.resource(inUse: .gliner)?.directory == state.modelFolder
        #endif
    }

    private func modelName(_ folder: URL) -> String {
        #if os(iOS)
            return folder.lastPathComponent
        #else
            return library?.artifactTitle(.gliner) ?? folder.lastPathComponent
        #endif
    }

    #if os(macOS)
        private var modelToChoose: URL? {
            guard let directory = library?.resource(inUse: .gliner)?.directory, state.modelFolder != directory,
                attempted != directory
            else { return nil }
            return directory
        }
    #endif
}
