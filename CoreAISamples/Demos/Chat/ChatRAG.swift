import Foundation
import PDFKit
import RAGContext
import SwiftUI

struct RAGSettings: Sendable {
    var web = false
    var local = false
    var searchResults = 8
    var braveAPIKey = ""
    var pagesToRead = 5
    var pageCharacterBudget = 3_000
    var sourceTokenBudget = RAGContext.sourceTokenBudget
    var documentsDirectory: URL = RAGRetriever.defaultDocumentsDirectory()

    /// The source budgets Chat offers (tokens). The first is the default.
    static let budgetChoices = [3_000, 6_000]

    var provider: String { braveAPIKey.isEmpty ? "DuckDuckGo" : "Brave" }

    @MainActor
    static func fromDefaults() -> RAGSettings {
        var settings = RAGSettings()
        settings.braveAPIKey = UserDefaults.standard.string(forKey: AgentViewModel.braveKeyDefaultsKey) ?? ""
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
    var sourceTokens = 0
    var seconds: Double = 0

    var citations: [String] { RAGContext.citations(sources) }

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

    static func gather(question: String, settings: RAGSettings, count: @Sendable (String) -> Int) async -> RAGRetrieval {
        let start = ContinuousClock.now
        var out = RAGRetrieval()
        var candidates: [RAGSource] = []

        if settings.local {
            let local = RAGDocuments.load(directory: settings.documentsDirectory) { PDFDocument(url: $0)?.string }
            if local.isEmpty {
                out.notes.append("Documents: no readable .txt, .md or .pdf file in \(settings.documentsDirectory.path(percentEncoded: false))")
            }
            candidates += local
        }

        if settings.web {
            candidates += await webCandidates(question: question, settings: settings, into: &out)
        }

        out.candidates = candidates.count
        let fixed = count(RAGContext.prompt(question: question, sources: [], today: RAGContext.today()))
        out.sources = RAGContext.select(
            candidates, question: question, budget: min(settings.sourceTokenBudget, RAGContext.promptTokenBudget - fixed),
            count: count, rank: bestRanking)
        out.sourceTokens = out.sources.map { count(RAGContext.measuredBlock($0)) }.reduce(0, +)
        out.seconds = (ContinuousClock.now - start) / .seconds(1)
        return out
    }

    static func bestRanking(_ passages: [String], query: String) -> [Int] {
        let ranked = AgentTools.bm25Ranking(passages, terms: AgentTools.matchTerms(query))
        return ranked.isEmpty ? RAGContext.rank(passages, query: query) : ranked
    }

    private static func webCandidates(question: String, settings: RAGSettings, into out: inout RAGRetrieval) async -> [RAGSource] {
        let results: [AgentTools.SearchResult]
        do {
            results = settings.braveAPIKey.isEmpty
                ? try await AgentTools.duckDuckGoResults(query: question, limit: settings.searchResults)
                : try await AgentTools.braveResults(query: question, limit: settings.searchResults, key: settings.braveAPIKey)
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

        let wanted = results
            .prefix(settings.pagesToRead)
            .compactMap { result -> (AgentTools.SearchResult, URL)? in
                guard let url = URL(string: result.url), let scheme = url.scheme?.lowercased(),
                    scheme == "http" || scheme == "https", !AgentTools.isLocalOrPrivate(url.host())
                else { return nil }
                return (result, url)
            }

        let budget = settings.pageCharacterBudget
        let read = await withTaskGroup(of: (Int, [RAGSource], String?).self) { group in
            for (index, pair) in wanted.enumerated() {
                group.addTask {
                    await passages(from: pair.0, url: pair.1, question: question, budget: budget)
                        .map { (index, $0, nil) } ?? (index, [], "Could not read \(pair.1.absoluteString)")
                }
            }
            var rows: [(Int, [RAGSource], String?)] = []
            for await row in group { rows.append(row) }
            return rows.sorted { $0.0 < $1.0 }
        }
        out.pagesRead = read.filter { !$0.1.isEmpty }.count
        out.notes += read.compactMap(\.2)
        return read.flatMap(\.1)
    }

    private static func passages(from result: AgentTools.SearchResult, url: URL, question: String, budget: Int) async -> [RAGSource]? {
        var html = ""
        var text = ""
        do {
            html = try await AgentTools.download(url: url)
            text = AgentTools.plainText(fromHTML: html)
        } catch {
            AgentTools.log("[rag] \(url.absoluteString) direct read failed: \(error)")
        }
        if text.count < 400 {
            guard let rendered = try? await AgentPageReader.readableText(url: url) else { return nil }
            text = AgentTools.condense(rendered)
        }
        guard !text.isEmpty else { return nil }

        let date = RAGContext.publishedDate(html: html)
            ?? RAGContext.isoDate(in: result.snippet)
            ?? RAGContext.isoDate(in: String(text.prefix(600)))
            ?? ""
        let relevant = AgentTools.pageText(text, query: question, budget: budget)
        let title = result.title.isEmpty ? (url.host() ?? url.absoluteString) : result.title
        return RAGContext.chunks(relevant).map {
            RAGSource(title: title, url: url.absoluteString, date: date, text: $0)
        }
    }
}

struct RAGControls: View {
    @Environment(ChatViewModel.self) private var vm

    var body: some View {
        @Bindable var vm = vm
        return HStack(spacing: 10) {
            Toggle("RAG", isOn: $vm.ragEnabled)
            if vm.ragEnabled {
                Toggle("Web", isOn: $vm.ragWeb)
                Toggle("Docs", isOn: $vm.ragLocal)
                Picker("Sources", selection: $vm.ragBudget) {
                    ForEach(RAGSettings.budgetChoices, id: \.self) { Text("\($0.formatted()) tok").tag($0) }
                }
                .pickerStyle(.menu)
                .fixedSize()
                .help("How many tokens of sources go into the prompt")
            }
        }
        .toggleStyle(.switch)
        .controlSize(.mini)
        .font(.caption)
        .disabled(vm.isResponding)
        .fixedSize()
    }
}

struct RAGSourceFooter: View {
    let notes: [String]
    let sources: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(notes, id: \.self) { line in
                Label(line, systemImage: "exclamationmark.triangle")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            if !sources.isEmpty {
                Text("Sources")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(sources, id: \.self) { line in
                    Text(line)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .textSelection(.enabled)
    }
}
