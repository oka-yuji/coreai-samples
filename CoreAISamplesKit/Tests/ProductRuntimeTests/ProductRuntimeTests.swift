import Foundation
import ProductCore
import Testing

@testable import ProductRuntime

struct ProductRuntimeTests {
    @Test("The bundled product labels cover the displayed fields in order")
    func schema() throws {
        #expect(try ProductLabel.bundled().map(\.field) == ProductField.allCases)
    }

    @Test("Network input is refused before attempting a non-official connection")
    func refusedAddress() async throws {
        let url = try #require(URL(string: "https://example.com/specs"))
        await #expect(throws: ProductError.unsupportedURL) {
            try await ProductPageClient().read(url: url, capturedAt: Date(timeIntervalSince1970: 0))
        }
    }
    @Test("A redirect to a non-official host is refused before contacting it")
    func rejectsRedirect() async throws {
        let calls = ProductHTTPCalls()
        let client = ProductPageClient { url in
            await calls.append(url)
            return ProductHTTPPage(
                status: 302, location: "https://example.com/elsewhere", mime: "text/html", data: Data())
        }
        let url = try #require(URL(string: "https://www.apple.com/specs"))
        await #expect(throws: ProductError.unsupportedURL) {
            try await client.read(url: url, capturedAt: Date(timeIntervalSince1970: 0))
        }
        #expect(await calls.count == 1)
    }

    @Test("An official relative redirect retains the final source and readable text")
    func followsOfficialRedirect() async throws {
        let client = ProductPageClient { url in
            if url.path == "/old" {
                return ProductHTTPPage(status: 302, location: "/new", mime: "text/html", data: Data())
            }
            return ProductHTTPPage(
                status: 200, location: nil, mime: "text/html", data: Data("<main>Memory 16 GB</main>".utf8))
        }
        let url = try #require(URL(string: "https://www.apple.com/old"))
        let source = try await client.read(url: url, capturedAt: Date(timeIntervalSince1970: 0))
        #expect(source.url?.path == "/new")
        #expect(source.text == "Memory 16 GB")
    }

    @Test("HTTP failures and unsupported formats never produce product cards")
    func rejectsHTTPFailures() async throws {
        let url = try #require(URL(string: "https://www.microsoft.com/specs"))
        let denied = ProductPageClient { _ in
            ProductHTTPPage(status: 403, location: nil, mime: "text/html", data: Data("Denied".utf8))
        }
        await #expect(throws: ProductError.http(403)) {
            try await denied.read(url: url, capturedAt: Date(timeIntervalSince1970: 0))
        }
        let pdf = ProductPageClient { _ in
            ProductHTTPPage(status: 200, location: nil, mime: "application/pdf", data: Data("PDF".utf8))
        }
        await #expect(throws: ProductError.unsupportedContent) {
            try await pdf.read(url: url, capturedAt: Date(timeIntervalSince1970: 0))
        }
    }

    @Test("Repeated redirects and oversized pages are bounded")
    func limitsResponses() async throws {
        let calls = ProductHTTPCalls()
        let loop = ProductPageClient { url in
            await calls.append(url)
            return ProductHTTPPage(status: 302, location: "/again", mime: "text/html", data: Data())
        }
        let url = try #require(URL(string: "https://www.apple.com/again"))
        await #expect(throws: ProductError.unsupportedURL) {
            try await loop.read(url: url, capturedAt: Date(timeIntervalSince1970: 0))
        }
        #expect(await calls.count == 4)
        let big = ProductPageClient { _ in
            ProductHTTPPage(
                status: 200, location: nil, mime: "text/plain",
                data: Data(repeating: 65, count: ProductPageClient.byteLimit + 1))
        }
        await #expect(throws: ProductError.pageTooLarge) {
            try await big.read(url: url, capturedAt: Date(timeIntervalSince1970: 0))
        }
    }

}

private actor ProductHTTPCalls {
    private(set) var count = 0
    func append(_ url: URL) { count += 1 }
}
