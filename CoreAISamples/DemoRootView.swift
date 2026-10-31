import DecideFeature
import ExtractFeature
import FeatureStore
import ImageSearchFeature
import Observation
import ProductFeature
import RecipeFeature
import SpeechFeature
import SwiftUI
import VideoSearchFeature

enum Demo: String, CaseIterable, Identifiable {
    case chat
    case decide
    case recipes
    case imageSearch
    case videoSearch
    case extract
    case speech
    case ocr
    case models

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chat: "Choose model"
        case .decide: "Clef-Flash 9B"
        case .recipes: "Clef-Flash 9B"
        case .imageSearch: "EmbeddingGemma 2"
        case .videoSearch: "EmbeddingGemma 2"
        case .extract: "GLiNER2.5-multi"
        case .speech: "Irodori"
        case .ocr: "TeleOCR"
        case .models: "Models"
        }
    }

    var summary: String {
        switch self {
        case .chat: "Talk, search the web and run multi-step tasks on a local model"
        case .decide: "Check a document against an editable checklist"
        case .recipes: "Can you make this recipe with what you have?"
        case .imageSearch: "Find selected images with your own words"
        case .videoSearch: "Find moments in a video with your own words"
        case .extract: "Compare product specifications from official descriptions"
        case .speech: "Generate and play speech from text with Irodori"
        case .ocr: "Read text and tables from a document image"
        case .models: "Get, open and prepare model bundles"
        }
    }

    var systemImage: String {
        switch self {
        case .chat: "bubble.left.and.bubble.right"
        case .decide: "checklist"
        case .recipes: "carrot"
        case .imageSearch: "photo.badge.magnifyingglass"
        case .videoSearch: "film.stack"
        case .extract: "text.magnifyingglass"
        case .speech: "waveform"
        case .ocr: "doc.viewfinder"
        case .models: "square.and.arrow.down"
        }
    }
}

struct DemoRootView: View {
    private let lifetime: ConversationLifetime
    @State private var windowID: UUID
    @Environment(\.scenePhase) private var scenePhase

    @AppStorage(DemoLanguage.preferenceKey) private var languageCode = DemoLanguage.initialCode
    @State private var navigator = DemoNavigator()
    @State private var chat: ChatViewModel
    @State private var host: ConversationHost
    @State private var models = ModelsViewModel()
    @State private var decide = DecideAssembly.makeStore()
    @State private var recipes = RecipeAssembly.makeStore()
    @State private var products = ProductAssembly.makeStore()
    @State private var imageSearch = ImageSearchAssembly.makeStore()
    @State private var videoSearch = VideoSearchAssembly.makeStore()
    @State private var speech = SpeechAssembly.makeStore()
    @State private var ocr = OCRMacLibrary()
    @State private var showingExtractionLab = false
    @State private var extract = Store<ExtractFeature>(
        initialState: ExtractState(
            form: ExtractResources.form, labels: ExtractResources.examples.labels,
            lastAssetPath: UserDefaults.standard.string(forKey: ExtractRunner.assetFolderKey),
            appVersion: ExtractRunner.appVersion),
        run: ExtractRunner(service: GLiNERExtractService(correctionWords: ExtractResources.correctionWords)).run)

    init(lifetime: ConversationLifetime) {
        self.lifetime = lifetime
        let id = UUID()
        _windowID = State(initialValue: id)
        let chat = ChatViewModel()
        _chat = State(initialValue: chat)
        _host = State(initialValue: ConversationHost(chat: chat, termination: { lifetime.finished(id, success: $0) }))
    }

    var body: some View {
        @Bindable var navigator = navigator
        NavigationSplitView {
            ConversationSidebar()
        } detail: {
            switch navigator.selection {
            case .chat?: ConversationView()
            case .decide?: DecideView(store: decide)
            case .recipes?: RecipeView(store: recipes)
            case .imageSearch?: ImageSearchView(store: imageSearch)
            case .videoSearch?: VideoSearchView(store: videoSearch)
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
            case .speech?: SpeechView(store: speech)
            case .ocr?: OCRHostView(library: ocr)
            case .models?: ModelsView()
            case nil: ContentUnavailableView("Select a demo", systemImage: "square.grid.2x2")
            }
        }
        .environment(navigator)
        .environment(chat)
        .environment(host)
        .environment(models)
        .environment(decide)
        .environment(speech)
        .environment(ocr)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("New conversation", systemImage: "square.and.pencil") {
                    host.startNew()
                    navigator.selection = .chat
                }
                .disabled(!host.canManageHistory)
                .help("Start a new conversation")
            }
            ToolbarItem(placement: .primaryAction) {
                DemoLanguagePicker(
                    code: Binding(
                        get: { (DemoLanguage(rawValue: languageCode) ?? .english).rawValue },
                        set: { languageCode = $0 }))
            }
        }
        .environment(\.locale, (DemoLanguage(rawValue: languageCode) ?? .english).locale)
        .onAppear {
            lifetime.register(windowID, host: host)
            if LaunchArguments.value("--gui-e2e") == nil {
                host.store.send(.restoreLatestRequested)
                host.preferences.searchKey.send(.loadRequested)
            } else {
                host.store.send(.library(.refresh))
            }
        }
        .onDisappear {
            ocr.close()
            lifetime.close(windowID)
        }
        .onChange(of: navigator.selection) { old, new in
            if old == .ocr && new != .ocr { ocr.leave() }
        }
        .onChange(of: scenePhase) { _, phase in ocr.enter(phase) }
        .onChange(of: scenePhase) { if scenePhase == .active { host.store.send(.library(.refresh)) } }
        .onChange(of: host.state.id) { host.restorePreferences() }
        .onChange(of: chat.loadStarted) { host.refreshIfModelChanged() }
        .task {
            await GUIDriver.shared.start(navigator: navigator, chat: chat, host: host, models: models, extract: extract)
        }
    }
}
