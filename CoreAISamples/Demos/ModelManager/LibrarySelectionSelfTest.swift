import CoreAIBackend
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryRuntime
import ModelLibraryStorage
import ModelLibraryTransfer
import ModelLibraryUI
import Observation
import SpeechFeature

@MainActor
enum LibrarySelectionSelfTest {
    static func checkPresentation(_ check: (String, Bool, String) -> Void) {
        do {
            let catalog = try MacDemoCatalog.make()
            let sidebarIDs = DemoModel.allCases.map(\.libraryID)
            check(
                "library/sidebarModels",
                sidebarIDs.count == Set(sidebarIDs).count
                    && Set(sidebarIDs) == Set(catalog.models.map(\.id)),
                "The actual sidebar model IDs match the Mac catalog.")
            let mapped = DemoModel.allCases.allSatisfy { model in
                guard let entry = catalog.model(model.libraryID) else { return false }
                let demos = model.demos.map(\.libraryID)
                return demos.count == Set(demos).count && Set(demos) == Set(entry.demos.map(\.demoID))
                    && demos.allSatisfy { catalog.demo($0)?.platforms.contains(.macOS) == true }
            }
            check(
                "library/sidebarDemos", mapped && Demo.allCases.allSatisfy { $0.model != nil },
                "Each actual sidebar demo belongs to its catalog model.")
            let spoken = SpokenSearchCatalog.selection
            check(
                "library/spokenSearchSidebar",
                Demo.spokenSearch.model == .embeddingGemma && Demo.spokenSearch.libraryID == spoken.demoID
                    && catalog.demo(spoken.demoID)?.allowsPreviewInteraction == true
                    && MacModelLibraryServices.assembly.registry.adapter(for: spoken.demoID) is SpokenSearchAdapter,
                "Recording search opens from the EmbeddingGemma 2 sidebar and can request the model for a topic search."
            )
            for code in ["en", "ja"] {
                let strings = DemoStrings(locale: Locale(identifier: code))
                let example = strings.text("Example word from the recording")
                let topicExample = strings.text("Example question about the recording")
                let labels = [
                    SpokenSearchCatalog.title, "Choose a recording…", "Word to find", "Stop playback", "Word", "Topic",
                    "Search mode", "Question or topic", "Finding related passages…",
                ]
                check(
                    "library/spokenSearchLanguage/" + code,
                    example != "Example word from the recording" && !example.isEmpty
                        && topicExample != "Example question about the recording" && !topicExample.isEmpty
                        && labels.allSatisfy { code == "en" ? strings.text($0) == $0 : strings.text($0) != $0 },
                    "The recording search title, both search modes, controls and examples use the requested display language."
                )
            }
            check(
                "library/chronosSidebarForecast",
                Demo.taxiForecast.model == .chronos
                    && DemoModel.chronos.demos == [.taxiForecast, .productSales, .seoulBike]
                    && Demo.taxiForecast.libraryID == ChronosDemoCatalog.selection.demoID,
                "Chronos opens fixed scenes and verified NYC API dates from its sidebar section.")
            check(
                "library/salesSidebarForecast",
                Demo.productSales.model == .chronos
                    && Demo.productSales.libraryID == ChronosDemoCatalog.salesSelection.demoID,
                "Product sales has its own sidebar destination and shares the Chronos model.")
            check(
                "library/bikeSidebarForecast",
                Demo.seoulBike.model == .chronos
                    && Demo.seoulBike.libraryID == ChronosDemoCatalog.bikeSelection.demoID,
                "Seoul Bike has its own sidebar destination and shares the Chronos model.")
            check(
                "library/voxcpm2SidebarSpeech",
                Demo.voiceDesign.model == .voxcpm2 && DemoModel.voxcpm2.demos == [.voiceDesign]
                    && Demo.voiceDesign.libraryID == "voice-design" && DemoModel.voxcpm2.libraryID == "voxcpm2"
                    && catalog.model("voxcpm2")?.platforms == [.macOS]
                    && MacModelLibraryServices.unconnectedDemoIDs.contains(Demo.voiceDesign.libraryID),
                "VoxCPM2 opens its own speech demo from its sidebar section through a usage lease.")
            let irodoriExamples = SpeechAssembly.makeStore().state.examples
            let voxcpm2Examples = SpeechAssembly.makeStore(.voxcpm2).state.examples
            check(
                "speech/voxcpm2Examples",
                voxcpm2Examples.map(\.id) == ["plain", "calm", "steady"] && !irodoriExamples.isEmpty
                    && irodoriExamples.allSatisfy { !voxcpm2Examples.contains($0) },
                "The VoxCPM2 screen loads its bundled examples and the Irodori screen keeps its own.")
            guard let model = catalog.model(ChronosDemoCatalog.selection.modelID) else {
                check("library/chronosPresentation", false, "The Chronos model is missing.")
                return
            }
            let state = ModelLibraryState(
                catalog: catalog, environment: LibraryEnvironment(platform: .macOS), owner: UUID())
            for code in ["en", "ja"] {
                let locale = Locale(identifier: code)
                let translated = Bundle.main.localizedString(
                    forKey: model.category, value: nil, table: nil,
                    localizations: [Locale.Language(identifier: code)])
                let classification = ModelClassificationPresentation.classification(for: model, locale: locale)
                let row = state.rowDetail(for: model, classification: classification, locale: locale)
                check(
                    "library/chronosCategory/" + code,
                    !translated.isEmpty && (code == "en" ? translated == model.category : translated != model.category)
                        && classification == translated && row.hasPrefix(translated + " · ")
                        && model.name == "Chronos-2" && model.category == "Time Series Forecasting",
                    "The UI classification and row use the main app catalog in the requested locale.")
            }
        } catch { check("library/presentation", false, error.localizedDescription) }
    }

    static func run(_ check: (String, Bool, String) -> Void) async {
        checkPresentation(check)
        await checkRecoveryPresentation(check)
        let root = URL.temporaryDirectory.appending(path: "library-selection-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            let catalog = try MacDemoCatalog.make()
            await checkSpokenSessions(catalog: catalog, check: check)
            let qwen = catalog.model("qwen3.8-27b")?.artifacts.first
            check(
                "library/conversationCatalog", qwen == (try LanguageModelArtifacts.qwen()), "Pinned Mac Qwen artifact")
            let assembly = ModelLibraryAssembly(
                adapters: [], store: ManagedModelStore(root: root.appending(path: "managed")),
                transfer: ArtifactTransfer())
            let library = assembly.makeLibrary(catalog: catalog, platform: .macOS, capabilities: [:], owner: UUID())
            library.store.send(.refresh)
            for await busy in Observations({ library.store.state.installations.values.contains { $0.phase.isBusy } })
            where !busy { break }
            check(
                "library/noUninstalledConversationModel",
                library.store.state.installations[LanguageModelArtifacts.qwenID]?.resource == nil,
                "A missing artifact is not a model-menu candidate.")
            let combinations = (0..<16).map { bits in
                ModelMenu.isDisabled(
                    loading: bits & 1 != 0, responding: bits & 2 != 0,
                    busy: bits & 4 != 0, running: bits & 8 != 0)
            }
            check(
                "switch/disabledWhileBusy", combinations == [false] + Array(repeating: true, count: 15),
                "All busy combinations")
            let first = try bundle("qwen35", modelID: "Qwen/Qwen3.5-0.8B", in: root)
            let other = try bundle("qwen38", modelID: "Qwen/Qwen3.8-27B", in: root)
            let wrong = try bundle("clef", modelID: "Cloudflare/clef-flash", in: root)
            let noSource = try bundle("unknown", modelID: nil, in: root)
            check(
                "bundles/supportedFamily", (try? BundleSupport.supportedBundle(at: first)) != nil, "Qwen is supported.")
            check(
                "bundles/rejectOtherFamily", (try? BundleSupport.supportedBundle(at: wrong)) == nil,
                "Clef is a decision model.")
            check(
                "bundles/rejectMissingSource", (try? BundleSupport.supportedBundle(at: noSource)) == nil,
                "A source identity is required.")
            let current = [first, URL(filePath: first.path), first.appending(path: ".")]
            check(
                "switch/currentLocation",
                current.allSatisfy { ModelSwitch.isLoaded($0, phase: .ready, current: first) }
                    && !ModelSwitch.isLoaded(other, phase: .ready, current: first)
                    && !ModelSwitch.isLoaded(first, phase: .empty, current: first), "Canonical location and ready state"
            )
            let preferences = ConversationPreferences(persists: false)
            preferences.resetForHeadless()
            let chat = ChatViewModel()
            let host = ConversationHost(chat: chat, preferences: preferences)
            let before = host.store
            chat.externallyBusy = true
            let refused = await ModelSwitch.choose(first, chat: chat, host: host)
            check(
                "switch/refusesWhileBusy", !refused && chat.phase == .empty && host.store === before,
                "Busy input is ignored.")
            chat.externallyBusy = false
            let switched = await ModelSwitch.choose(first, chat: chat, host: host)
            check(
                "switch/preparesConversationBeforeLoad", switched && host.store !== before && chat.bundleURL == first,
                "The malformed tokenizer prevents native inference after switch preparation.")
            let managed = ChatViewModel(managed: true)
            await managed.load(first)
            let rejected: Bool
            if case .failed(let message) = managed.phase {
                rejected = message == ManagedModelStoreError.notInstalled.localizedDescription
            } else {
                rejected = false
            }
            check(
                "library/refusesUnmanagedRuntimePath", rejected && managed.model == nil,
                "Normal model loading requires the managed artifact and its runtime lease.")
        } catch { check("library/modelSelection", false, error.localizedDescription) }
    }

    private static func checkSpokenSessions(catalog: ModelCatalog, check: (String, Bool, String) -> Void) async {
        let selection = SpokenSearchCatalog.selection
        guard let artifact = catalog.model(selection.modelID)?.artifacts.first, let revision = artifact.sourceRevision
        else {
            check("spokenSearch/sessionOwners", false, "The embedding model is missing from the catalog.")
            return
        }
        let resource = ValidatedModelResource(
            modelID: selection.modelID, artifactID: artifact.id, contract: artifact.contract, revision: revision,
            upstreamRevision: artifact.upstream?.revision, directory: .temporaryDirectory,
            sizeBytes: artifact.sizeBytes ?? 0)
        let owner = UUID()
        let first = ModelDemoSession(
            id: LibraryTicket(owner: owner, number: 1), selection: selection, resource: resource)
        let next = ModelDemoSession(
            id: LibraryTicket(owner: owner, number: 2), selection: selection, resource: resource)
        let other = ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: 1), selection: selection, resource: resource)
        let firstGate = DemoExecutionGate(session: first) { true }
        let nextGate = DemoExecutionGate(session: next) { true }
        let otherGate = DemoExecutionGate(session: other) { true }
        let adapter = SpokenSearchAdapter()
        guard let original = adapter.bind(first, gate: firstGate),
            let independent = adapter.bind(other, gate: otherGate)
        else {
            check("spokenSearch/sessionOwners", false, "Two independent spoken word search sessions could not bind.")
            return
        }
        original.store.send(.appeared)
        independent.store.send(.appeared)
        let reused = adapter.bind(first, gate: firstGate) === original
        let refusesMismatch = adapter.bind(next, gate: nextGate) == nil && adapter.bind(first, gate: otherGate) == nil
        await adapter.stop(next)
        let wrongStopPreserved = original.store.state.isActive
        await adapter.stop(first)
        let retired = adapter.bind(first, gate: firstGate) == nil
        guard let replacement = adapter.bind(next, gate: nextGate) else {
            check("spokenSearch/sessionOwners", false, "A new ticket could not bind after the first session stopped.")
            return
        }
        replacement.store.send(.appeared)
        await adapter.stop(first)
        let revoked = DemoExecutionGate(session: next) { false }
        check(
            "spokenSearch/sessionOwners",
            original !== independent && reused && refusesMismatch && wrongStopPreserved && retired
                && !original.store.state.isActive && replacement !== original && replacement.store.state.isActive
                && adapter.bind(next, gate: revoked) == nil,
            "Spoken word search binds the matching ticket, joins stop and rejects retired tickets and revoked gates.")
        await adapter.stop(next)
        check(
            "spokenSearch/independentWindows",
            independent.store.state.isActive && adapter.bind(other, gate: otherGate) === independent
                && adapter.bind(first, gate: firstGate) == nil && adapter.bind(next, gate: nextGate) == nil,
            "Closing and reopening one window keeps the recording session in the other window active.")
        await adapter.stop(other)
    }

    private static func checkRecoveryPresentation(_ check: (String, Bool, String) -> Void) async {
        let chat = ChatViewModel()
        let url = URL.temporaryDirectory.appending(path: UUID().uuidString)
        let key = ModelStorage.key(url)
        ModelStorage.prepared.insert(key)
        defer { ModelStorage.prepared.remove(key) }
        await chat.load(url) { _, _, first in
            let loading = chat.phase == .loading(preparing: false)
            first()
            await drainMainQueue()
            check(
                "library/currentRepair", loading && chat.phase == .loading(preparing: true),
                "A known model changes from loading to preparing when recovery begins.")
            chat.clearModel()
            await chat.load(url) { _, _, _ in
                first()
                await drainMainQueue()
                check(
                    "library/staleRepair", chat.phase == .loading(preparing: false),
                    "A notification for the previous load cannot change the next load.")
                throw PreparedCopyError.recoveryFailed
            }
            first()
            await drainMainQueue()
            throw CancellationError()
        }
        let expected = ChatViewModel.preparedCopyMessage(for: PreparedCopyError.recoveryFailed) ?? ""
        check(
            "library/terminalRepair", chat.phase == .failed(expected) && expected.contains("Clear Prepared Copy"),
            "The actionable failure stays visible after stale callbacks and completion.")
        for language in ["en", "ja"] {
            let strings = DemoStrings(locale: Locale(identifier: language))
            let detail = strings.text(expected)
            check(
                "library/terminalRepair/" + language,
                detail.contains(strings.text("Clear Prepared Copy")) && (language == "en" || detail != expected),
                "The terminal message names the visible clear button in the selected language.")
        }
    }

    private static func drainMainQueue() async {
        await withCheckedContinuation { continuation in
            DispatchQueue.main.async { continuation.resume() }
        }
    }

    private static func bundle(_ name: String, modelID: String?, in root: URL) throws -> URL {
        let url = root.appending(path: name, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: url.appending(path: "tokenizer"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: url.appending(path: name + ".aimodel"), withIntermediateDirectories: true)
        try Data("{".utf8).write(to: url.appending(path: "tokenizer/tokenizer.json"))
        var metadata: [String: Any] = [
            "name": name, "assets": ["main": name + ".aimodel"],
            "language": ["vocab_size": 16, "max_context_length": 64],
        ]
        if let modelID { metadata["source"] = ["model_definition": "torch", "hf_model_id": modelID] }
        try JSONSerialization.data(withJSONObject: metadata).write(to: url.appending(path: "metadata.json"))
        return url
    }
}
