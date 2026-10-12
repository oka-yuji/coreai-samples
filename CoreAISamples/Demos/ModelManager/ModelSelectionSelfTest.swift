import Foundation

enum ModelSelectionSelfTest {
    static let otherFamilyReason =
        "This sample runs Qwen3, Qwen3.5 and Qwen3.8 bundles made with Apple's converter. "
        + "clef was converted from Cloudflare/clef-flash."
    static let missingIDReason =
        "metadata.json has no source.hf_model_id, so this sample cannot tell which model family the bundle is."

    @MainActor
    static func run(_ check: (String, Bool, String) -> Void) async {
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
        } catch {
            check("bundles/fixtures", false, "\(error)")
        }
    }

    @MainActor
    private static func bundleChecks(
        qwen: [URL], clef: URL, unknown: URL, defaults: UserDefaults, _ check: (String, Bool, String) -> Void
    ) async {
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
                && models.folders[1].name == nil && models.folders[1].problem == otherFamilyReason,
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
