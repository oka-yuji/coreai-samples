import Foundation
import ModelLibraryFeature
import ModelLibraryRuntime
import ModelLibraryStorage
import ModelLibraryTransfer
import Observation

@MainActor
enum LibrarySelectionSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        let root = URL.temporaryDirectory.appending(path: "library-selection-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            let catalog = try MacDemoCatalog.make()
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
