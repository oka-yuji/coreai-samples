import FeatureStore
import Foundation
import ProductCore
import ProductFeature
import SwiftUI

struct ProductComparisonContent: View {
    let store: Store<ProductFeature>

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            ProductEntry(store: store) {}
            Text("\(store.state.cards.count) / \(ProductState.cardLimit) cards")
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            ProductComparisonCards(store: store)
        }
    }
}

struct ProductComparisonPlaceholder: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Picker("Description source", selection: .constant(ProductDraft.Mode.url)) {
                Text("Official page").tag(ProductDraft.Mode.url)
                Text("Paste description").tag(ProductDraft.Mode.text)
            }
            .pickerStyle(.segmented)
            TextField("Apple or Microsoft product URL", text: .constant(""))
                .textFieldStyle(.roundedBorder)
            Button("Create comparison card", systemImage: "plus.rectangle.on.rectangle") {}
                .buttonStyle(.borderedProminent)
        }
        .disabled(true)
    }
}

struct ProductEntry: View {
    let store: Store<ProductFeature>
    let paste: () -> Void

    private var state: ProductState { store.state }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            form
            ProductStatus(store: store, paste: paste)
            if let source = state.importedSource {
                DisclosureGroup("Read imported text") {
                    Text(verbatim: source.text).textSelection(.enabled).font(.callout)
                }
            }
        }
    }

    private var form: some View {
        VStack(alignment: .leading, spacing: 14) {
            Picker(
                "Description source",
                selection: Binding(get: { state.draft.mode }, set: { mode in store.changeDraft { $0.mode = mode } })
            ) {
                Text("Official page").tag(ProductDraft.Mode.url)
                Text("Paste description").tag(ProductDraft.Mode.text)
            }
            .pickerStyle(.segmented).frame(maxWidth: 360)
            if state.draft.mode == .url {
                TextField(
                    "Apple or Microsoft product URL",
                    text: Binding(
                        get: { state.draft.address }, set: { value in store.changeDraft { $0.address = value } })
                )
                .textFieldStyle(.roundedBorder)
                .onSubmit { store.send(.submit(Date())) }
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
                    text: Binding(get: { state.draft.title }, set: { value in store.changeDraft { $0.title = value } })
                )
                .textFieldStyle(.roundedBorder)
                TextEditor(
                    text: Binding(get: { state.draft.text }, set: { value in store.changeDraft { $0.text = value } })
                )
                .frame(minHeight: 110, maxHeight: 180)
                .padding(6)
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.primary.opacity(0.12)))
                .accessibilityLabel("Product description")
                Text("Paste the product description, including any options and conditions.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 12) {
                Button("Create comparison card", systemImage: "plus.rectangle.on.rectangle") {
                    store.send(.submit(Date()))
                }
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

    private func setAddress(_ address: String) { store.changeDraft { $0.address = address } }
}

struct ProductStatus: View {
    let store: Store<ProductFeature>
    let paste: () -> Void

    var body: some View {
        switch store.state.phase {
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
                if store.state.draft.mode == .url {
                    Button("Paste the description instead") {
                        store.changeDraft { $0.mode = .text }
                        paste()
                    }
                }
            }
            .foregroundStyle(.secondary)
        case .idle: EmptyView()
        }
    }
}

struct ProductComparisonCards: View {
    let store: Store<ProductFeature>
    @State private var inspection: ProductInspection?

    var body: some View {
        Group {
            if store.state.cards.isEmpty {
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
                            ForEach(store.state.cards) { card in
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
        .sheet(item: $inspection) { ProductSourceView(inspection: $0) }
    }
}

extension Store where F == ProductFeature {
    func changeDraft(_ update: (inout ProductDraft) -> Void) {
        var draft = state.draft
        update(&draft)
        send(.draftChanged(draft))
    }
}
