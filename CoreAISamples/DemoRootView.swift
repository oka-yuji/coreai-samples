import DecideFeature
import ExtractFeature
import FeatureStore
import ImageSearchFeature
import ModelLibraryFeature
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

enum DemoModel: CaseIterable {
    case qwen
    case clefFlash
    case embeddingGemma
    case gliner
    case irodori
    case teleOCR

    var title: String {
        switch self {
        case .qwen: "Qwen3.8 27B"
        case .clefFlash: "Clef-Flash 9B"
        case .embeddingGemma: "EmbeddingGemma 2"
        case .gliner: "GLiNER2.5-multi"
        case .irodori: "Irodori"
        case .teleOCR: "TeleOCR"
        }
    }

    var libraryID: String {
        switch self {
        case .qwen: "qwen3.8-27b"
        case .clefFlash: "clef-flash-9b"
        case .embeddingGemma: "embeddinggemma2"
        case .gliner: "gliner2.5-multi"
        case .irodori: "irodori"
        case .teleOCR: "teleocr"
        }
    }

    var demos: [Demo] { Demo.allCases.filter { $0.model == self } }
}

extension Demo {
    var model: DemoModel? {
        switch self {
        case .chat: .qwen
        case .decide, .recipes: .clefFlash
        case .imageSearch, .videoSearch: .embeddingGemma
        case .extract: .gliner
        case .speech: .irodori
        case .ocr: .teleOCR
        case .models: nil
        }
    }
}

struct DemoRootView: View {
    private let lifetime: ConversationLifetime
    @State private var windowID: UUID
    @Environment(\.scenePhase) private var scenePhase

    @AppStorage(DemoLanguage.preferenceKey) private var languageCode = DemoLanguage.initialCode
    @State private var navigator: DemoNavigator
    @State private var chat: ChatViewModel
    @State private var host: ConversationHost
    @State private var models = ModelsViewModel()
    @State private var decide: Store<DecideFeature>
    @State private var recipes: Store<RecipeFeature>
    @State private var products = ProductAssembly.makeStore()
    @State private var imageSearch = ImageSearchAssembly.makeStore()
    @State private var videoSearch = VideoSearchAssembly.makeStore()
    @State private var speech = SpeechAssembly.makeStore()
    @State private var library: MacModelLibrary
    @State private var choice: MacModelChoice
    @State private var showingExtractionLab = false
    @State private var extract = Store<ExtractFeature>(
        initialState: ExtractState(
            form: ExtractResources.form, labels: ExtractResources.examples.labels, appVersion: ExtractRunner.appVersion),
        run: ExtractRunner(service: GLiNERExtractService(correctionWords: ExtractResources.correctionWords)).run)

    init(lifetime: ConversationLifetime) {
        self.lifetime = lifetime
        let id = UUID()
        _windowID = State(initialValue: id)
        let navigator = DemoNavigator()
        _navigator = State(initialValue: navigator)
        let choice = MacModelChoice(initialState: MacModelSelection.State()) { _, _ in }
        _choice = State(initialValue: choice)
        let library = MacModelLibrary()
        _library = State(initialValue: library)
        let managed = LaunchArguments.value("--gui-e2e") == nil
        let chat = ChatViewModel(managed: managed)
        let decide = DecideAssembly.makeStore(managed: managed)
        let recipes = RecipeAssembly.makeStore(managed: managed)
        _chat = State(initialValue: chat)
        _decide = State(initialValue: decide)
        _recipes = State(initialValue: recipes)
        _host = State(
            initialValue: ConversationHost(
                chat: chat, termination: { lifetime.finished(id, success: $0) },
                resumeModels: {
                    if navigator.selection == .decide { decide.send(.appeared) }
                    if navigator.selection == .recipes { recipes.send(.appeared) }
                },
                modelUnloaded: {
                    choice.send(.reset)
                    library.stopUsing(.qwen)
                },
                releaseModels: {
                    decide.send(.ended)
                    recipes.send(.ended)
                    for await phases in Observations({ (decide.state.phase, recipes.state.phase) }) {
                        if phases.0 != .closing && phases.1 != .closing { break }
                    }
                }))
    }

    private var modelLoadID: String? {
        guard choice.state.selecting, let resource = library.resource(inUse: .qwen) else { return nil }
        return "\(choice.state.request):" + ModelStorage.key(resource.directory)
    }

    var body: some View {
        lifetimeContent
            .onChange(of: host.state.id) { host.restorePreferences() }
            .onChange(of: chat.loadStarted) { host.refreshIfModelChanged() }
            .onChange(of: library.store?.state.usages[DemoModel.qwen.libraryID]) { _, usage in
                guard choice.state.selecting, !chat.isLoading, usage?.isActive != true, usage?.resource == nil else {
                    return
                }
                choice.send(.reset)
                library.stopUsing(.qwen)
            }
            .task(id: modelLoadID) {
                guard choice.state.selecting, let resource = library.resource(inUse: .qwen) else { return }
                let request = choice.state.request
                _ = await ModelSwitch.choose(resource.directory, chat: chat, host: host)
                guard choice.state.request == request else { return }
                if chat.phase != .ready { library.stopUsing(.qwen) }
                choice.send(.selected(request))
            }
            .task(id: choice.state.unloading ? choice.state.request : nil) {
                guard choice.state.unloading else { return }
                let request = choice.state.request
                guard host.state.canLeave, !chat.isLoading else {
                    choice.send(.unloaded(request))
                    return
                }
                host.runner.session.releaseModel()
                chat.clearModel()
                await chat.loader.release()
                guard choice.state.request == request else { return }
                library.stopUsing(.qwen)
                choice.send(.unloaded(request))
            }
            .task {
                await GUIDriver.shared.start(
                    navigator: navigator, chat: chat, host: host, models: models, extract: extract)
            }
    }

    private var lifetimeContent: some View {
        content
            .onAppear {
                lifetime.register(windowID, host: host)
                if LaunchArguments.value("--gui-e2e") == nil {
                    host.store.send(.restoreLatestRequested)
                    host.preferences.removeLegacySearchKey()
                } else {
                    host.store.send(.library(.refresh))
                }
            }
            .onDisappear {
                library.close()
                lifetime.close(windowID)
            }
            .onChange(of: navigator.selection) { old, new in
                if old == .ocr && new != .ocr { library.leave() }
            }
            .onChange(of: library.installed(.irodori)?.directory) { old, new in
                if old != nil, new != old { speech.send(.bundleUnavailable) }
            }
            .onChange(of: library.installed(.gliner)?.directory) { old, new in
                guard old != nil, new != old else { return }
                products.send(.modelUnavailable)
                extract.send(.closed)
            }
            .onChange(of: library.installed(.embeddingGemma)?.directory) { old, new in
                guard old != nil, new != old, ImageSearchAssembly.launchBundle == nil else { return }
                imageSearch.send(.disappeared)
                videoSearch.send(.disappeared)
            }
            .onChange(of: library.installed(.clefFlash)?.directory) { old, new in
                guard old != nil, new != old else { return }
                decide.send(.ended)
                recipes.send(.ended)
            }
            .onChange(of: scenePhase) { _, phase in library.enter(phase) }
            .onChange(of: scenePhase) { if scenePhase == .active { host.store.send(.library(.refresh)) } }
    }

    @ViewBuilder private var content: some View {
        @Bindable var navigator = navigator
        NavigationSplitView {
            ConversationSidebar()
        } detail: {
            switch navigator.selection {
            case .chat?: ConversationView()
            case .decide?: DecideView(store: decide)
            case .recipes?: RecipeView(store: recipes)
            case .imageSearch?:
                ImageSearchView(store: imageSearch).environment(ImageSearchAssembly.launchBundle == nil ? library : nil)
            case .videoSearch?:
                VideoSearchView(store: videoSearch).environment(ImageSearchAssembly.launchBundle == nil ? library : nil)
            case .extract?:
                if LaunchArguments.value("--gui-e2e") != nil && LaunchArguments.value("--extract") != nil {
                    ExtractView(store: extract).environment(nil as MacModelLibrary?)
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
            case .ocr?: OCRHostView(library: library)
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
        .environment(library)
        .environment(choice)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                DemoLanguagePicker(
                    code: Binding(
                        get: { (DemoLanguage(rawValue: languageCode) ?? .english).rawValue },
                        set: { languageCode = $0 }))
            }
        }
        .environment(\.locale, (DemoLanguage(rawValue: languageCode) ?? .english).locale)
    }
}
