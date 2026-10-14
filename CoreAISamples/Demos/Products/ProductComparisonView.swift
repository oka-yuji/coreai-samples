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
    @State private var inspection: ProductInspection?
    @Environment(\.locale) private var locale

    private var state: ProductState { store.state }
    private var strings: DemoStrings { DemoStrings(locale: locale) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header
                status
                comparison
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
                    VStack(alignment: .leading, spacing: 18) {
                        sourceEntry
                        status
                        if let source = state.importedSource {
                            DisclosureGroup("Read imported text") {
                                Text(verbatim: source.text).textSelection(.enabled).font(.callout)
                            }
                        }
                    }.padding(20)
                }
                .demoNavigationTitle("Add a product")
                .toolbar { Button("Done") { addingProduct = false } }
            }
            .frame(idealWidth: 660, idealHeight: 430)
        }
        .onChange(of: state.cards.count) { oldCount, newCount in
            if newCount > oldCount { addingProduct = false }
        }
        .sheet(item: $inspection) { ProductSourceView(inspection: $0) }
        .demoNavigationTitle("Product notes")
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Compare product descriptions").font(.title2.bold())
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

    private var sourceEntry: some View {
        VStack(alignment: .leading, spacing: 14) {
            Picker(
                "Description source",
                selection: Binding(get: { state.draft.mode }, set: { mode in changeDraft { $0.mode = mode } })
            ) {
                Text("Official page").tag(ProductDraft.Mode.url)
                Text("Paste description").tag(ProductDraft.Mode.text)
            }
            .pickerStyle(.segmented).frame(maxWidth: 360)
            if state.draft.mode == .url {
                TextField(
                    "Apple or Microsoft product URL",
                    text: Binding(get: { state.draft.address }, set: { value in changeDraft { $0.address = value } })
                )
                .textFieldStyle(.roundedBorder)
                .onSubmit { addCard() }
                HStack(spacing: 10) {
                    Text("Try a page:").foregroundStyle(.secondary)
                    Button("MacBook Air") { setAddress("https://www.apple.com/jp/macbook-air/specs/") }
                    Button("Surface Laptop 5") {
                        setAddress(
                            "https://support.microsoft.com/ja-jp/surface/models/surface-laptop-5-specs-and-features")
                    }
                }
                .font(.caption)
            } else {
                TextField(
                    "Product name (optional)",
                    text: Binding(get: { state.draft.title }, set: { value in changeDraft { $0.title = value } })
                )
                .textFieldStyle(.roundedBorder)
                TextEditor(text: Binding(get: { state.draft.text }, set: { value in changeDraft { $0.text = value } }))
                    .frame(minHeight: 110, maxHeight: 180)
                    .padding(6)
                    .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.primary.opacity(0.12)))
                    .accessibilityLabel("Product description")
                Text("Paste the product description, including any options and conditions.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 12) {
                Button("Create comparison card", systemImage: "plus.rectangle.on.rectangle") { addCard() }
                    .buttonStyle(.borderedProminent).disabled(!state.canAdd)
                if state.busy {
                    Button("Stop") { store.send(.stop) }.disabled(state.phase == .stopping)
                }
                Spacer()
            }
        }
        .padding(18)
        .background(.background, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Color.primary.opacity(0.08)))
        .disabled(state.preview)
    }

    @ViewBuilder private var status: some View {
        switch state.phase {
        case .importing:
            HStack {
                ProgressView("Reading the product description…")
                Spacer()
                Button("Stop") { store.send(.stop) }
            }
        case .extracting:
            HStack {
                ProgressView("Finding specifications on this device…")
                Spacer()
                Button("Stop") { store.send(.stop) }
            }
        case .stopping: ProgressView("Finishing the current request…")
        case .failed(let message):
            VStack(alignment: .leading, spacing: 6) {
                Label("Could not create this card", systemImage: "exclamationmark.circle")
                RuntimeMessageText(message).font(.callout)
                if state.draft.mode == .url {
                    Button("Paste the description instead") {
                        changeDraft { $0.mode = .text }
                        addingProduct = true
                    }
                }
            }
            .foregroundStyle(.secondary)
        case .idle: EmptyView()
        }
    }

    @ViewBuilder private var comparison: some View {
        if state.cards.isEmpty {
            ContentUnavailableView {
                Label("Start with two products", systemImage: "rectangle.split.2x1")
            } description: {
                Text("Add a page or paste a description. Cards line up by specification so you can compare them.")
            }
            .frame(minHeight: 220)
        } else {
            VStack(alignment: .leading, spacing: 12) {
                Text("Values mentioned on each page").font(.headline)
                Text("A page may list several configurations. Open a value to check its original context.")
                    .font(.caption).foregroundStyle(.secondary)
                ScrollView(.horizontal) {
                    HStack(alignment: .top, spacing: 16) {
                        ForEach(state.cards) { card in
                            ProductCardView(
                                card: card,
                                inspect: { field in
                                    inspection = ProductInspection(id: card.id, memo: card.memo, field: field)
                                }, remove: { store.send(.remove(card.id)) }
                            )
                            .frame(width: 300)
                        }
                    }
                    .padding(.bottom, 8)
                }
            }
        }
    }

    private func changeDraft(_ update: (inout ProductDraft) -> Void) {
        var draft = state.draft
        update(&draft)
        store.send(.draftChanged(draft))
    }

    private func setAddress(_ address: String) { changeDraft { $0.address = address } }
    private func addCard() { store.send(.submit(Date())) }
}
