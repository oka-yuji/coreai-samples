import FeatureStore
import Foundation
import ProductCore
import ProductFeature
import ProductRuntime
import SwiftUI

@MainActor
enum ProductAssembly {
    static func makeStore() -> Store<ProductFeature> {
        let saved = UserDefaults.standard.string(forKey: ExtractRunner.assetFolderKey)
        let defaultFolder = ModelStorage.root.appending(path: "gliner/gliner25_multi_fp32_t256_t512")
        let folder: URL? =
            if let saved, !saved.isEmpty {
                URL(filePath: saved)
            } else if FileManager.default.fileExists(atPath: defaultFolder.appending(path: "contract.json").path) {
                defaultFolder
            } else { nil }
        return Store(
            initialState: ProductState(modelFolder: folder), run: ProductRunner(service: LiveProductService()).run)
    }

    static func previewStore() -> Store<ProductFeature> {
        var state = ProductState()
        state.preview = true
        do {
            guard let path = Bundle.main.url(forResource: "product-preview", withExtension: "json") else {
                throw ProductError.invalidSchema
            }
            let records = try JSONDecoder().decode([PreviewRecord].self, from: Data(contentsOf: path))
            for (index, record) in records.enumerated() {
                let source = try ProductSource(
                    title: record.title, text: record.text, url: nil,
                    capturedAt: Date(timeIntervalSince1970: 1_791_936_000))
                let scalars = source.text.unicodeScalars
                let candidates = try record.values.map { value in
                    guard let range = source.text.range(of: value.text) else { throw ProductError.invalidEvidence }
                    return ProductCandidate(
                        field: value.field,
                        range: scalars.distance(
                            from: scalars.startIndex, to: range.lowerBound)..<scalars.distance(
                                from: scalars.startIndex, to: range.upperBound),
                        text: value.text, probability: 0.5)
                }
                state.cards.append(
                    ProductCard(id: index, memo: try ProductMemo(source: source, candidates: candidates)))
            }
        } catch { state.phase = .failed(error.localizedDescription) }
        return Store(initialState: state) { _, _ in }
    }

    private struct PreviewRecord: Decodable {
        struct Value: Decodable {
            let field: ProductField
            let text: String
        }
        let title: String
        let text: String
        let values: [Value]
    }
}

struct ProductPreviewRoot: View {
    @State private var store = ProductAssembly.previewStore()
    @AppStorage(DemoLanguage.preferenceKey) private var languageCode = DemoLanguage.initialCode

    var body: some View {
        NavigationStack {
            ProductComparisonView(store: store)
                .toolbar { DemoLanguagePicker(code: $languageCode) }
        }
        .environment(\.locale, (DemoLanguage(rawValue: languageCode) ?? .english).locale)
    }
}
