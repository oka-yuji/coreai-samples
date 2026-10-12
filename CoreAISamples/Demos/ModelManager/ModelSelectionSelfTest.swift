import Foundation

enum ModelSelectionSelfTest {
    static let otherFamilyReason =
        "This sample runs Qwen3, Qwen3.5 and Qwen3.8 bundles made with Apple's converter. "
        + "clef was converted from Cloudflare/clef-flash."
    static let missingIDReason =
        "metadata.json has no source.hf_model_id, so this sample cannot tell which model family the bundle is."

    @MainActor
    static func run(_ check: (String, Bool, String) -> Void) async {
        let published = Catalog.models.first
        check(
            "catalog/qwen38-27b-4bit",
            Catalog.models.count == 1 && published?.id == "qwen3-8-27b-4bit"
                && published?.repo == "okayuji/Qwen3.8-27B-coreai-4bit"
                && published?.revision == "c034c9857455a48cea715fec19eb49b3227b2458"
                && published?.sizeBytes == 16_852_520_981,
            "\(Catalog.models.map { "\($0.id) \($0.repo)@\($0.revision) \($0.sizeBytes)" })")
        let name = "model-selection-\(UUID().uuidString)"
        let root = FileManager.default.temporaryDirectory.appending(path: name)
        guard let defaults = UserDefaults(suiteName: name) else {
            check("bundles/isolatedDefaults", false, "could not create isolated defaults")
            return
        }
        defer {
            defaults.removePersistentDomain(forName: name)
            try? FileManager.default.removeItem(at: root)
        }
        do {
            let qwen = try [
                ("qwen3", "Qwen/Qwen3-0.6B"), ("qwen35", "Qwen/Qwen3.5-0.8B"), ("qwen38", "Qwen/Qwen3.8-27B"),
            ]
            .map { try bundle($0.0, modelID: $0.1, in: root) }
            let clef = try bundle("clef", modelID: "Cloudflare/clef-flash", in: root)
            let unknown = try bundle("unknown", modelID: nil, in: root)
            await bundleChecks(qwen: qwen, clef: clef, unknown: unknown, defaults: defaults, check)
            await menuChecks(qwen: qwen, clef: clef, root: root, defaults: defaults, check)
        } catch {
            check("bundles/fixtures", false, "\(error)")
        }
    }

    @MainActor
    private static func bundleChecks(
        qwen: [URL], clef: URL, unknown: URL, defaults: UserDefaults, _ check: (String, Bool, String) -> Void
    ) async {
        check(
            "bundles/foldersKeyUnchanged", ModelsViewModel.foldersDefaultsKey == "localFolders",
            ModelsViewModel.foldersDefaultsKey)
        let accepted = qwen.filter { (try? BundleSupport.supportedBundle(at: $0)) != nil }
        check("bundles/supportedPrefixes", accepted == qwen, "accepted \(accepted.map(\.lastPathComponent))")

        let models = ModelsViewModel(defaults: defaults)
        func stored() -> [String] { defaults.stringArray(forKey: ModelsViewModel.foldersDefaultsKey) ?? [] }
        var registered: [String] = []
        for url in qwen {
            models.open(url)
            if models.message == nil, stored().contains(ModelStorage.key(url)) {
                registered.append(url.lastPathComponent)
            }
        }
        check("bundles/openRegistersSupported", registered.count == qwen.count, "registered \(registered)")

        for (name, url, reason) in [
            ("bundles/openRefusesOtherFamily", clef, otherFamilyReason),
            ("bundles/openRefusesMissingID", unknown, missingIDReason),
        ] {
            models.message = nil
            models.open(url)
            check(
                name, models.message == reason && !stored().contains(ModelStorage.key(url)),
                models.message ?? "nil")
        }

        defaults.set([ModelStorage.key(qwen[1]), ModelStorage.key(clef)], forKey: ModelsViewModel.foldersDefaultsKey)
        models.refresh()
        let rows = models.folders.map { "\($0.name ?? "nil")|\($0.problem ?? "nil")" }
        check(
            "bundles/refreshShowsReason",
            models.folders.count == 2 && models.folders[0].name == "qwen35" && models.folders[0].problem == nil
                && models.folders[1].name != nil && models.folders[1].problem == otherFamilyReason
                && !models.folders[1].canLoad && models.folders[1].canClear,
            "\(rows)")

        var refusedLoads: [String] = []
        for url in qwen {
            let chat = ChatViewModel()
            await chat.load(url)
            if case .failed(let message) = chat.phase, chat.bundleURL == url,
                message != otherFamilyReason, message != missingIDReason
            {
                continue
            }
            refusedLoads.append("\(url.lastPathComponent): \(chat.phase)")
        }
        check("bundles/loadPassesSupported", refusedLoads.isEmpty, "\(refusedLoads)")

        for (name, url, reason) in [
            ("bundles/loadRefusesOtherFamily", clef, otherFamilyReason),
            ("bundles/loadRefusesMissingID", unknown, missingIDReason),
        ] {
            let chat = ChatViewModel()
            await chat.load(url)
            check(name, chat.phase == .failed(reason) && chat.model == nil, "\(chat.phase)")
        }
    }

    @MainActor
    private static func menuChecks(
        qwen: [URL], clef: URL, root: URL, defaults: UserDefaults, _ check: (String, Bool, String) -> Void
    ) async {
        let missing = root.appending(path: "missing", directoryHint: .isDirectory)
        let catalog = ["a", "b"].map {
            CatalogModel(
                id: $0, name: "Catalog \($0)", repo: "selftest/model-\($0)",
                revision: String(repeating: "0", count: 40),
                sizeBytes: 1, memory: "1 GB", platforms: "macOS 27")
        }
        func describe(_ entries: [ModelsViewModel.MenuEntry]) -> [String] {
            entries.map { "\($0.name)|\($0.id)|\($0.isCurrent)" }
        }
        let key = ModelsViewModel.foldersDefaultsKey
        defaults.set([ModelStorage.key(qwen[1]), ModelStorage.key(clef), ModelStorage.key(missing)], forKey: key)
        let models = ModelsViewModel(catalog: catalog, defaults: defaults)
        models.rows[0].isDownloaded = true
        let download = models.rows[0].directory
        let byFolder = models.menuEntries(current: qwen[1])
        check(
            "menu/listsDownloadedAndReadable",
            describe(byFolder.downloaded) == ["Catalog a|\(ModelStorage.key(download))|false"]
                && describe(byFolder.onThisMac) == ["qwen35|\(ModelStorage.key(qwen[1]))|true"],
            "\(describe(byFolder.downloaded)) \(describe(byFolder.onThisMac))")
        let byDownload = models.menuEntries(current: download)
        check(
            "menu/marksCurrentDownload",
            byDownload.downloaded.map(\.isCurrent) == [true] && byDownload.onThisMac.map(\.isCurrent) == [false],
            "\(describe(byDownload.downloaded)) \(describe(byDownload.onThisMac))")
        defaults.set([ModelStorage.key(clef), ModelStorage.key(missing)], forKey: key)
        let empty = ModelsViewModel(catalog: catalog, defaults: defaults).menuEntries(current: nil)
        check(
            "menu/emptyWithoutCandidates", empty.downloaded.isEmpty && empty.onThisMac.isEmpty,
            "\(describe(empty.downloaded)) \(describe(empty.onThisMac))")

        let agent = AgentViewModel()
        agent.persistsSettings = false
        let earlier = AgentViewModel.Entry(kind: .user, title: "You", text: "earlier task")
        agent.entries = [earlier]
        let idle = ChatViewModel()
        let current = ModelsViewModel.MenuEntry(name: "qwen35", url: qwen[1], isCurrent: true)
        await ModelMenu.choose(current, chat: idle, agent: agent)
        check(
            "menu/chooseCurrentDoesNothing",
            idle.phase == .empty && idle.bundleURL == nil && agent.entries.map(\.id) == [earlier.id],
            "\(idle.phase) agentEntries=\(agent.entries.count)")

        let chat = ChatViewModel()
        let before = chat.loadStarted
        let other = ModelsViewModel.MenuEntry(name: "qwen3", url: qwen[0], isCurrent: false)
        await ModelMenu.choose(other, chat: chat, agent: agent)
        var reachedRuntime = false
        if case .failed(let message) = chat.phase {
            reachedRuntime = message != otherFamilyReason && message != missingIDReason
        }
        check(
            "menu/chooseStartsLoadAndNewConversation",
            chat.bundleURL == qwen[0] && chat.loadStarted > before && chat.entries.isEmpty && agent.entries.isEmpty
                && reachedRuntime,
            "\(chat.phase) agentEntries=\(agent.entries.count)")

        let states = [
            (false, false, false, false), (true, false, false, false), (false, true, false, false),
            (false, false, true, false), (false, false, false, true),
        ]
        let disabled = states.map {
            ModelMenu.isDisabled(loading: $0.0, responding: $0.1, busy: $0.2, agentRunning: $0.3)
        }
        check("menu/disabledWhileBusy", disabled == [false, true, true, true, true], "\(disabled)")
    }

    static func bundle(_ name: String, modelID: String?, in root: URL) throws -> URL {
        let url = root.appending(path: name, directoryHint: .isDirectory)
        let manager = FileManager.default
        try manager.createDirectory(at: url.appending(path: "tokenizer"), withIntermediateDirectories: true)
        try manager.createDirectory(at: url.appending(path: "\(name).aimodel"), withIntermediateDirectories: true)
        try Data("{".utf8).write(to: url.appending(path: "tokenizer/tokenizer.json"))
        var metadata: [String: Any] = [
            "name": name, "assets": ["main": "\(name).aimodel"],
            "language": ["vocab_size": 16, "max_context_length": 64],
        ]
        if let modelID { metadata["source"] = ["model_definition": "torch", "hf_model_id": modelID] }
        try JSONSerialization.data(withJSONObject: metadata).write(to: url.appending(path: "metadata.json"))
        return url
    }
}
