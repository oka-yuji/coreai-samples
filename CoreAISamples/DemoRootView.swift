import DecideFeature
import ExtractFeature
import FeatureStore
import ImageSearchFeature
import Observation
import ProductFeature
import RecipeFeature
import SwiftUI

enum Demo: String, CaseIterable, Identifiable {
    case chat
    case decide
    case recipes
    case imageSearch
    case extract
    case models

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chat: "Chat"
        case .decide: "Decide"
        case .recipes: "Recipes"
        case .imageSearch: "Image search"
        case .extract: "Extract"
        case .models: "Models"
        }
    }

    var summary: String {
        switch self {
        case .chat: "Talk, search the web and run multi-step tasks on a local model"
        case .decide: "Check a document against an editable checklist"
        case .recipes: "Can you make this recipe with what you have?"
        case .imageSearch: "Find selected images with your own words"
        case .extract: "Compare product specifications from official descriptions"
        case .models: "Get, open and prepare model bundles"
        }
    }

    var systemImage: String {
        switch self {
        case .chat: "bubble.left.and.bubble.right"
        case .decide: "checklist"
        case .recipes: "carrot"
        case .imageSearch: "photo.badge.magnifyingglass"
        case .extract: "text.magnifyingglass"
        case .models: "square.and.arrow.down"
        }
    }
}

struct DemoRootView: View {
    @AppStorage(DemoLanguage.preferenceKey) private var languageCode = DemoLanguage.initialCode
    @State private var navigator = DemoNavigator()
    @State private var chat: ChatViewModel
    @State private var host: ConversationHost
    @State private var models = ModelsViewModel()
    @State private var decide = DecideAssembly.makeStore()
    @State private var recipes = RecipeAssembly.makeStore()
    @State private var products = ProductAssembly.makeStore()
    @State private var imageSearch = ImageSearchAssembly.makeStore()
    @State private var showingExtractionLab = false
    @State private var extract = Store<ExtractFeature>(
        initialState: ExtractState(
            form: ExtractResources.form, labels: ExtractResources.examples.labels,
            lastAssetPath: UserDefaults.standard.string(forKey: ExtractRunner.assetFolderKey),
            appVersion: ExtractRunner.appVersion),
        run: ExtractRunner(service: GLiNERExtractService(correctionWords: ExtractResources.correctionWords)).run)

    init() {
        let chat = ChatViewModel()
        _chat = State(initialValue: chat)
        _host = State(initialValue: ConversationHost(chat: chat))
    }

    var body: some View {
        @Bindable var navigator = navigator
        NavigationSplitView {
            List(selection: $navigator.selection) {
                ForEach(Demo.allCases) { demo in
                    NavigationLink(value: demo) {
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(LocalizedStringKey(demo.title))
                                Text(LocalizedStringKey(demo.summary))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        } icon: {
                            Image(systemName: demo.systemImage)
                        }
                    }
                }
            }
            .demoNavigationTitle("Demos")
            .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        } detail: {
            switch navigator.selection {
            case .chat?: ConversationView()
            case .decide?: DecideView(store: decide)
            case .recipes?: RecipeView(store: recipes)
            case .imageSearch?: ImageSearchView(store: imageSearch)
            case .extract?:
                if LaunchArguments.value("--gui-e2e") != nil && LaunchArguments.value("--extract") != nil {
                    ExtractView(store: extract)
                } else {
                    ProductComparisonView(store: products)
                        .toolbar { Button("Extraction lab") { showingExtractionLab = true } }
                        .sheet(isPresented: $showingExtractionLab) {
                            NavigationStack {
                                ExtractView(store: extract).toolbar { Button("Done") { showingExtractionLab = false } }
                            }
                            .frame(minWidth: 820, minHeight: 640)
                        }
                }
            case .models?: ModelsView()
            case nil: ContentUnavailableView("Select a demo", systemImage: "square.grid.2x2")
            }
        }
        .environment(navigator)
        .environment(chat)
        .environment(host)
        .environment(models)
        .environment(decide)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                DemoLanguagePicker(
                    code: Binding(
                        get: { (DemoLanguage(rawValue: languageCode) ?? .english).rawValue },
                        set: { languageCode = $0 }))
            }
        }
        .environment(\.locale, (DemoLanguage(rawValue: languageCode) ?? .english).locale)
        .onChange(of: chat.loadStarted) { host.refreshIfModelChanged() }
        .task {
            await GUIDriver.shared.start(navigator: navigator, chat: chat, host: host, models: models, extract: extract)
        }
    }
}
