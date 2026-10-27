#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ModelLibraryUI
    import SwiftUI

    struct PhoneModelLibraryScreen: View {
        @AppStorage(DemoLanguage.preferenceKey) private var languageCode = DemoLanguage.initialCode
        @Environment(\.scenePhase) private var scenePhase
        @State private var library: Result<ModelLibraryAssembly.Library, any Error>

        init() {
            _library = State(
                initialValue: Result {
                    let definitions = try IOSModelLibraryAssembly.catalog()
                    let translations = DemoLanguage.allCases.map { DemoStrings(locale: $0.locale) }
                    var terms: [String: [String]] = [:]
                    for demo in definitions.demos {
                        terms[demo.id] = translations.flatMap { [$0.text(demo.title), $0.text(demo.summary)] }
                    }
                    for model in definitions.models {
                        terms[model.id] = translations.map { $0.text(model.category) }
                    }
                    return IOSModelLibraryAssembly.make(
                        catalog: try IOSModelLibraryAssembly.catalog(searchTerms: terms))
                })
        }

        var body: some View {
            switch library {
            case .success(let library):
                ModelLibraryView(
                    store: library.store, renderer: library.rendering,
                    leadingToolbar: AnyView(DemoLanguagePicker(code: $languageCode))
                )
                .environment(\.locale, (DemoLanguage(rawValue: languageCode) ?? .english).locale)
                .onChange(of: scenePhase) { _, phase in IOSModelLibraryAssembly.enter(phase, library: library.store) }
                .onDisappear { IOSModelLibraryAssembly.close(library.store) }
            case .failure:
                ContentUnavailableView("Could not open Models", systemImage: "exclamationmark.triangle")
            }
        }
    }
#endif
