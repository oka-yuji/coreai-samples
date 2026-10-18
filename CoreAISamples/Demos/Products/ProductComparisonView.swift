import FeatureStore
import Foundation
import ProductCore
import ProductFeature
import SwiftUI
import UniformTypeIdentifiers

struct ProductComparisonView: View {
    let store: Store<ProductFeature>
    var assetFolders: [URL] = []
    @State private var importingModel = false
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
        .onAppear { store.send(.appeared) }
        .onDisappear { store.send(.closed) }
        .fileImporter(isPresented: $importingModel, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { store.send(.modelChosen(url)) }
        }
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
                                || state.modelFolder == nil)
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
                    if let folder = state.modelFolder {
                        Text(verbatim: folder.lastPathComponent).lineLimit(1).truncationMode(.middle)
                    } else {
                        Text("Choose a GLiNER model folder to begin")
                    }
                    Spacer()
                    if !assetFolders.isEmpty {
                        Menu("Model folder") {
                            ForEach(assetFolders, id: \.self) { url in
                                Button(url.lastPathComponent) { store.send(.modelChosen(url)) }
                            }
                        }
                    } else {
                        Button("Model folder", systemImage: "folder") { importingModel = true }
                    }
                }
                .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
