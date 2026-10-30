import Foundation
import Testing

@testable import coreai_verify

@Suite("read_page tool folder boundary")
struct PageToolTests {
    private func folders() throws -> (root: URL, pages: URL) {
        let root = FileManager.default.temporaryDirectory.appending(path: "page-tool-" + UUID().uuidString)
        let pages = root.appending(path: "pages", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: pages, withIntermediateDirectories: true)
        try Data("handbook index".utf8).write(to: pages.appending(path: "index.txt"))
        try Data("outside secret".utf8).write(to: root.appending(path: "secret.txt"))
        return (root, pages)
    }

    @Test("A page listed in the folder is returned in full")
    func readsListedPage() async throws {
        let (root, pages) = try folders()
        defer { try? FileManager.default.removeItem(at: root) }
        let tool = PageTool(folder: pages.path(percentEncoded: false), log: ToolLog(snapshot: nil))
        #expect(try await tool.call(arguments: .init(name: "index")) == "handbook index")
    }

    @Test("A page name that leaves the folder is refused without reading the outside file")
    func refusesEscapingNames() async throws {
        let (root, pages) = try folders()
        defer { try? FileManager.default.removeItem(at: root) }
        let tool = PageTool(folder: pages.path(percentEncoded: false), log: ToolLog(snapshot: nil))
        for name in ["../secret", "./../secret", root.appending(path: "secret").path(percentEncoded: false)] {
            let reply = try await tool.call(arguments: .init(name: name))
            #expect(!reply.contains("outside secret"))
            #expect(reply == "No page named \(name). Available pages: index.")
        }
    }
}
