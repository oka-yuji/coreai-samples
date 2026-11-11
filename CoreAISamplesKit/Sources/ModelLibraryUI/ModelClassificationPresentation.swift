import Foundation
import ModelLibraryFeature

public enum ModelClassificationPresentation {
    public static func classification(
        for model: LibraryModel, locale: Locale, bundle: Bundle = .main
    ) -> String {
        classification(for: model, locale: locale) { key, locale in
            bundle.localizedString(forKey: key, value: nil, table: nil, localizations: [locale.language])
        }
    }

    static func classification(
        for model: LibraryModel, locale: Locale, resolve: (String, Locale) -> String
    ) -> String {
        let category = model.category.isEmpty ? "" : resolve(model.category, locale)
        return [category, model.architecture].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
    }
}
