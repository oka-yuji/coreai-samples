import Foundation
import RAGContext
import Synchronization

@MainActor
enum RAGStructureSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        await pageBranches(check)
        let raw =
            "<main><h1>Manual</h1><h2>Alpha</h2><p>" + String(repeating: "Alpha uses ten minutes. ", count: 25)
            + "</p></main>"
        let text = AgentTools.plainText(fromHTML: raw)
        let page = RAGPage(raw: raw, text: text, finalURL: "https://rag.example/final", format: .html)
        let inputs = RAGInputs(
            local: { _ in [] },
            search: { _, _ in [.init(title: "Manual", url: "https://rag.example/start", snippet: "Alpha")] },
            read: { _ in page })
        var settings = RAGSettings()
        settings.web = true
        let baseline = await RAGRetriever.gather(
            question: "Alpha", settings: settings, count: { $0.count }, inputs: inputs)
        settings.captureStructure = true
        let result = await RAGRetriever.gather(
            question: "Alpha", settings: settings, count: { $0.count }, inputs: inputs)
        check(
            "rag-structure/acquisition-preserves-flat", baseline.sources == result.sources,
            "baseline=\(baseline.sources.count) structured=\(result.sources.count)")
        check(
            "rag-structure/acquisition-metadata",
            result.documents.first?.structure?.isAvailable == true
                && result.documents.first?.finalURL == page.finalURL
                && result.documents.first?.version == RAGContext.fingerprint(text),
            "documents=\(result.documents.count)")
        await cancelled(settings: settings, inputs: inputs, check: check)
        let directory = URL.temporaryDirectory.appending(path: "rag-structure-local-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data("# Local\n## Alpha\nTen minutes.".utf8).write(to: directory.appending(path: "manual.md"))
            let old = RAGDocuments.load(directory: directory)
            var local = RAGSettings()
            local.local = true
            local.captureStructure = true
            local.documentsDirectory = directory
            let loaded = await RAGRetriever.gather(question: "Alpha", settings: local, count: { $0.count })
            check(
                "rag-structure/local-equivalence",
                loaded.sources == old && loaded.documents.first?.structure?.isAvailable == true,
                "sources=\(loaded.sources.count)")
        } catch { check("rag-structure/local-equivalence", false, error.localizedDescription) }
    }

    private static func pageBranches(_ check: (String, Bool, String) -> Void) async {
        guard let url = URL(string: "https://public.example/start"),
            let final = URL(string: "https://public.example/final")
        else { return }
        let policy = AgentFetchPolicy(resolve: { _ in [[8, 8, 8, 8]] })
        let raw = "<h1>Title</h1><p>" + String(repeating: "Long text. ", count: 50) + "</p>"
        do {
            let page = try await RAGRetriever.readPageDetails(
                url: url, policy: policy, download: { _ in (raw, final, "text/html") },
                render: { _ in throw AgentToolMessage(text: "unexpected render") })
            check(
                "rag-structure/direct-page",
                page.format == .html && page.finalURL == final.absoluteString
                    && page.text.utf8.elementsEqual(AgentTools.plainText(fromHTML: raw).utf8), page.finalURL)
            let rendered = try await RAGRetriever.readPageDetails(
                url: url, policy: policy, download: { _ in ("short", final, "text/plain") },
                render: { _ in ("Rendered text", final) })
            check(
                "rag-structure/rendered-page",
                rendered.format == .renderedText && rendered.text == "Rendered text"
                    && rendered.finalURL == final.absoluteString, rendered.text)
        } catch { check("rag-structure/page-branches", false, error.localizedDescription) }
        do {
            _ = try await RAGRetriever.readPageDetails(
                url: url, policy: policy,
                download: { _ in throw AgentToolMessage(text: "refused", forbidsPageFallback: true) },
                render: { _ in throw AgentToolMessage(text: "unexpected render") })
            check("rag-structure/refusal", false, "Unexpected success")
        } catch let failure as AgentToolMessage {
            check("rag-structure/refusal", failure.text == "refused", failure.text)
        } catch { check("rag-structure/refusal", false, error.localizedDescription) }
    }

    private static func cancelled(
        settings: RAGSettings, inputs: RAGInputs, check: (String, Bool, String) -> Void
    ) async {
        let begun = AsyncStream<Void>.makeStream()
        let release = AsyncStream<Void>.makeStream()
        let received = Mutex<RAGRetrieval?>(nil)
        var delayed = inputs
        delayed.read = { _ in
            begun.continuation.yield(())
            for await _ in release.stream { break }
            return RAGPage(raw: "<h1>Late</h1>", text: "Late", finalURL: "https://rag.example/late", format: .html)
        }
        let captured = delayed
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                let value = await RAGRetriever.gather(
                    question: "Alpha", settings: settings, count: { $0.count }, inputs: captured)
                received.withLock { $0 = value }
            }
            for await _ in begun.stream { break }
            group.cancelAll()
            release.continuation.yield(())
            release.continuation.finish()
        }
        begun.continuation.finish()
        let result = received.withLock { $0 }
        check(
            "rag-structure/cancelled-acquisition", result?.documents.isEmpty == true && result?.sources.isEmpty == true,
            "late structure and sources are discarded")
    }
}
