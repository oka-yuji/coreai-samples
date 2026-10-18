#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import ModelLibraryUI
    import SwiftUI

    struct PhoneModelLibraryScreen: View {
        @AppStorage(DemoLanguage.preferenceKey) private var languageCode = DemoLanguage.initialCode
        @State private var library: Result<Store<ModelLibraryFeature>, any Error>

        init() {
            _library = State(
                initialValue: Result {
                    let definitions = try IOSDemoCatalog.make()
                    let translations = DemoLanguage.allCases.map { DemoStrings(locale: $0.locale) }
                    var terms: [String: [String]] = [:]
                    for demo in definitions.demos {
                        terms[demo.id] = translations.flatMap { [$0.text(demo.title), $0.text(demo.summary)] }
                    }
                    for model in definitions.models {
                        terms[model.id] = translations.map { $0.text(model.category) }
                    }
                    let catalog = try IOSDemoCatalog.make(searchTerms: terms)
                    return Store(initialState: ModelLibraryState(catalog: catalog, platform: .iOS)) { effect, send in
                        ModelLibraryBrowsingRunner.run(effect, send: send)
                    }
                })
        }

        var body: some View {
            switch library {
            case .success(let store):
                ModelLibraryView(
                    store: store, renderer: PhoneDemoDescriptionRenderer(),
                    leadingToolbar: AnyView(DemoLanguagePicker(code: $languageCode))
                )
                .environment(\.locale, (DemoLanguage(rawValue: languageCode) ?? .english).locale)
            case .failure:
                ContentUnavailableView("Could not open Models", systemImage: "exclamationmark.triangle")
            }
        }
    }

    @MainActor
    private struct PhoneDemoDescriptionRenderer: ModelDemoRendering {
        func content(for session: ModelDemoSession) -> AnyView {
            AnyView(Text("Demo execution is not connected in this build."))
        }

        func preview(for selection: DemoSelection) -> AnyView {
            let labels: (input: String, action: String) =
                switch selection.demoID {
                case "comparison": ("Product description", "Compare")
                case "extraction": ("Text to extract", "Extract")
                case "inspection": ("Inspection answer", "Start inspection")
                case "image-search": ("Search query", "Search images")
                case "object-finder": ("What to find", "Start camera")
                default: ("Input", "Run demo")
                }
            return AnyView(
                VStack(alignment: .leading, spacing: 12) {
                    Text(LocalizedStringKey(labels.input)).font(.headline)
                    TextField("", text: .constant(""), axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .accessibilityLabel(Text(LocalizedStringKey(labels.input)))
                    Button {
                    } label: {
                        Text(LocalizedStringKey(labels.action))
                    }
                    .buttonStyle(.borderedProminent)
                }
                .disabled(true))
        }
    }
#endif
