import ConversationFeature
import Foundation
import RAGContext
import RAGDocumentRuntime
import SwiftUI

struct RAGSettings: Sendable {
    var web = false
    var local = false
    var searchResults = 8
    var braveAPIKey = ""
    var pagesToRead = 5
    var pageCharacterBudget = 3_000
    var sourceTokenBudget = RAGContext.sourceTokenBudget
    var today = RAGContext.today()
    var captureStructure = false
    var recordsDiagnostics = false
    var strategy = RAGStrategy.flat
    var selectionLimits = RAGSectionLimits()
    var documentsDirectory: URL = RAGRetriever.defaultDocumentsDirectory()

    static let budgetChoices = [3_000, 6_000]

    var provider: String { braveAPIKey.isEmpty ? "DuckDuckGo" : "Brave" }

    @MainActor
    static func fromDefaults() -> RAGSettings {
        var settings = RAGSettings()
        settings.captureStructure = LaunchArguments.flag("--rag-structure")
        settings.strategy = LaunchArguments.value("--rag-strategy").flatMap(RAGStrategy.init(rawValue:)) ?? .flat
        settings.braveAPIKey = UserDefaults.standard.string(forKey: ConversationPreferences.braveKeyKey) ?? ""
        if let path = LaunchArguments.value("--rag-docs") {
            settings.documentsDirectory = URL(filePath: path, directoryHint: .isDirectory)
        }
        return settings
    }
}

struct RAGRetrieval: Sendable {
    var sources: [RAGSource] = []
    var notes: [String] = []
    var searchResults = 0
    var pagesRead = 0
    var candidates = 0
    var candidatePassages: [RAGSource] = []
    var documents: [RAGDocumentSnapshot] = []
    var bodySources: [RAGSource] = []
    var selectionCandidates: [RAGCandidate] = []
    var atomicGroups: [[RAGSource]]?
    var requestedStrategy = RAGStrategy.flat.rawValue
    var effectiveStrategy = RAGStrategy.flat.rawValue
    var fallbacks: [String] = []
    var selectionLimits = RAGSectionLimits()
    var selectionTrace: RAGSectionTrace?
    var selectionFailure: String?
    var sourceTokens = 0
    var seconds: Double = 0

    func conversationSources() throws -> [ConversationSource] {
        try Self.convert(sources)
    }

    static func convert(_ sources: [RAGSource]) throws -> [ConversationSource] {
        try sources.map { source in
            guard let version = source.version, let offset = source.offset, let document = source.document else {
                throw ConversationSources.Failure.invalid
            }
            return ConversationSource(
                document: document, version: version, title: source.title, date: source.date,
                passages: [SourcePassage(offset: offset, text: source.text)])
        }
    }

    func prepare(
        using ledger: ConversationSources, question: String, today: String, budget: Int,
        count: (String) -> Int, mode: ConversationMode
    ) throws -> ConversationSources.Prepared? {
        if let atomicGroups {
            return try ledger.prepare(
                question: question, groups: atomicGroups.map(Self.convert), today: today,
                budget: budget, count: count, mode: mode)
        }
        return try ledger.prepare(
            question: question, sources: conversationSources(), today: today,
            budget: budget, count: count, mode: mode)
    }

    func suppliedSources(_ prepared: ConversationSources.Prepared) -> [RAGSource] {
        if let groupCount = prepared.groupCount, let atomicGroups {
            return RAGStructuredSelection.uniqueSources(atomicGroups.prefix(groupCount).flatMap { $0 })
        }
        return Array(sources.prefix(prepared.sources.count))
    }

    var summary: String {
        "search \(searchResults) / pages \(pagesRead) / passages \(candidates) → \(sources.count) sources, \(sourceTokens) tokens"
    }
}

enum RAGRetriever {
    static func defaultDocumentsDirectory() -> URL {
        let url = URL.documentsDirectory.appending(path: "rag", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @concurrent static func gather(
        question: String, settings: RAGSettings, count: @Sendable (String) -> Int,
        inputs: RAGInputs = .live
    ) async
        -> RAGRetrieval
    {
        let start = ContinuousClock.now
        var settings = inputs.configure(settings)
        if settings.strategy != .flat { settings.captureStructure = true }
        var out = RAGRetrieval()
        out.selectionLimits = settings.selectionLimits
        var candidates: [RAGSource] = []
        guard !Task.isCancelled else { return RAGRetrieval() }

        if settings.local {
            let local = inputs.local(settings.documentsDirectory)
            if local.isEmpty {
                out.notes.append(
                    "Documents: no readable .txt, .md or .pdf file in \(settings.documentsDirectory.path(percentEncoded: false))"
                )
            }
            let documents = local.map { settings.captureStructure ? RAGStructureReader.read($0) : $0 }
            if settings.captureStructure || settings.recordsDiagnostics { out.documents += documents }
            candidates += documents.flatMap(\.sources)
        }

        guard !Task.isCancelled else { return RAGRetrieval() }
        if settings.web {
            candidates += await webCandidates(question: question, settings: settings, inputs: inputs, into: &out)
        }

        guard !Task.isCancelled else { return RAGRetrieval() }
        out.candidates = candidates.count
        if settings.recordsDiagnostics { out.candidatePassages = candidates }
        let fixed = count(RAGContext.prompt(question: question, sources: [], today: settings.today))
        out.sources = RAGContext.select(
            candidates, question: question,
            budget: min(settings.sourceTokenBudget, RAGContext.promptTokenBudget - fixed),
            count: count, rank: bestRanking)
        out.bodySources = out.sources
        applyStructured(
            to: &out, question: question, settings: settings,
            budget: min(settings.sourceTokenBudget, RAGContext.promptTokenBudget - fixed), count: count)
        out.sourceTokens = out.sources.map { count(RAGContext.measuredBlock($0)) }.reduce(0, +)
        out.seconds = (ContinuousClock.now - start) / .seconds(1)
        return out
    }

    static func bestRanking(_ passages: [String], query: String) -> [Int] {
        let ranked = AgentTools.bm25Ranking(passages, terms: AgentTools.matchTerms(query))
        return ranked.isEmpty ? RAGContext.rank(passages, query: query) : ranked
    }

    private static func webCandidates(
        question: String, settings: RAGSettings, inputs: RAGInputs, into out: inout RAGRetrieval
    ) async
        -> [RAGSource]
    {
        let results: [AgentTools.SearchResult]
        do {
            results = try await inputs.search(question, settings)
        } catch let failure as AgentToolMessage {
            out.notes.append("Web search: \(failure.text)")
            return []
        } catch {
            out.notes.append("Web search: network error — \(error.localizedDescription)")
            return []
        }
        out.searchResults = results.count
        guard !results.isEmpty else {
            out.notes.append("Web search: no results for \"\(question)\"")
            return []
        }

        let wanted =
            results
            .prefix(settings.pagesToRead)
            .compactMap { result -> (AgentTools.SearchResult, URL)? in
                guard let url = URL(string: result.url), let scheme = url.scheme?.lowercased(),
                    scheme == "https", url.host != nil
                else { return nil }
                return (result, url)
            }

        let budget = settings.pageCharacterBudget
        let capture = settings.captureStructure
        let retain = capture || settings.recordsDiagnostics
        let read = await withTaskGroup(of: (Int, [RAGSource], RAGDocumentSnapshot?, String?).self) { group in
            for (index, pair) in wanted.enumerated() {
                group.addTask {
                    if let result = await passages(
                        from: pair.0, url: pair.1, question: question, budget: budget,
                        capture: capture, read: inputs.read)
                    {
                        return (index, result.sources, retain ? result.document : nil, nil)
                    }
                    return (index, [], nil, "Could not read \(pair.1.absoluteString)")
                }
            }
            var rows: [(Int, [RAGSource], RAGDocumentSnapshot?, String?)] = []
            for await row in group { rows.append(row) }
            return rows.sorted { $0.0 < $1.0 }
        }
        out.pagesRead = read.filter { !$0.1.isEmpty }.count
        out.notes += read.compactMap(\.3)
        out.documents += read.compactMap(\.2)
        return read.flatMap(\.1)
    }

    static func readPage(
        url: URL,
        policy: AgentFetchPolicy = AgentFetchPolicy(),
        download: @Sendable (URL) async throws -> String = AgentTools.download,
        render: @Sendable (URL) async throws -> String = AgentPageReader.readableText
    ) async throws -> (html: String, text: String) {
        try await policy.validate(url)
        var html = ""
        var text = ""
        do {
            html = try await download(url)
            text = AgentTools.plainText(fromHTML: html)
        } catch let failure as AgentToolMessage where failure.forbidsPageFallback {
            throw failure
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            AgentTools.log("[rag] \(url.absoluteString) direct read failed: \(error)")
        }
        if text.count < 400 {
            text = AgentTools.condense(try await render(url))
        }
        return (html, text)
    }

    static func passages(
        from result: AgentTools.SearchResult, url: URL, question: String, budget: Int, capture: Bool,
        read: @Sendable (URL) async throws -> RAGPage
    ) async -> (sources: [RAGSource], document: RAGDocumentSnapshot)? {
        let page: RAGPage
        do { page = try await read(url) } catch {
            AgentTools.log("[rag] \(url.absoluteString) read failed: \(error)")
            return nil
        }
        guard !page.text.isEmpty, !Task.isCancelled else { return nil }
        let date =
            RAGContext.publishedDate(html: page.raw)
            ?? RAGContext.isoDate(in: result.snippet)
            ?? RAGContext.isoDate(in: String(page.text.prefix(600))) ?? ""
        let title = result.title.isEmpty ? (url.host() ?? url.absoluteString) : result.title
        let acquired = RAGDocumentSnapshot(
            id: url.absoluteString, url: url.absoluteString, finalURL: page.finalURL, title: title,
            date: date, raw: page.raw, text: page.text, format: page.format)
        let document = capture ? RAGStructureReader.read(acquired) : acquired
        guard !Task.isCancelled else { return nil }
        let sources = RAGContext.select(
            document.sources, question: question, budget: budget,
            count: { $0.count }, rank: bestRanking)
        return (sources, document)
    }

}

struct RAGControls: View {
    @Environment(ChatViewModel.self) private var vm

    var body: some View {
        @Bindable var vm = vm
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Toggle("Search and sources", isOn: $vm.ragEnabled)
                if vm.ragEnabled {
                    Toggle("Web", isOn: $vm.ragWeb)
                    Toggle("Docs", isOn: $vm.ragLocal)
                        .help("Search .txt, .md and .pdf files in ~/Documents/rag.")
                    Picker("Source budget", selection: $vm.ragBudget) {
                        ForEach(RAGSettings.budgetChoices, id: \.self) { Text("\($0.formatted()) tok").tag($0) }
                    }
                    .pickerStyle(.menu)
                    .fixedSize()
                    .help("How many tokens of sources go into the prompt")
                }
            }
            Text(
                "Off: answer from the model and conversation without new searches. On: retrieve the selected sources before answering. Work can still save notes."
            )
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            if !vm.hasValidSourceSelection {
                Label("Choose Web or Docs to use sources.", systemImage: "info.circle")
                    .foregroundStyle(.secondary)
            }
        }
        .toggleStyle(.switch)
        .controlSize(.mini)
        .font(.caption)
        .disabled(vm.isResponding)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
