import Foundation
import ProductCore

struct ProductHTTPPage: Sendable {
    let status: Int
    let location: String?
    let mime: String
    let data: Data
}

public actor ProductPageClient {
    static let byteLimit = 2 * 1_048_576
    private let get: @Sendable (URL) async throws -> ProductHTTPPage

    public init() {
        let client = ProductHTTPClient()
        get = { url in try await client.get(url) }
    }

    init(get: @escaping @Sendable (URL) async throws -> ProductHTTPPage) {
        self.get = get
    }

    public func read(url: URL, capturedAt: Date) async throws -> ProductSource {
        var current = try ProductPage.validatedURL(url.absoluteString)
        for _ in 0..<4 {
            try Task.checkCancellation()
            let response = try await get(current)
            try Task.checkCancellation()
            guard response.data.count <= Self.byteLimit else { throw ProductError.pageTooLarge }
            if (300..<400).contains(response.status) {
                guard let location = response.location, let next = URL(string: location, relativeTo: current) else {
                    throw ProductError.unsupportedURL
                }
                current = try ProductPage.validatedURL(next.absoluteURL.absoluteString)
                continue
            }
            guard (200..<300).contains(response.status) else { throw ProductError.http(response.status) }
            guard ["text/html", "text/plain", "application/xhtml+xml"].contains(response.mime.lowercased()),
                let text = String(data: response.data, encoding: .utf8)
            else { throw ProductError.unsupportedContent }
            if response.mime.lowercased() == "text/plain" {
                return try ProductSource(
                    title: current.host() ?? "Product page", text: text, url: current, capturedAt: capturedAt)
            }
            return try ProductPage.source(html: text, url: current, capturedAt: capturedAt)
        }
        throw ProductError.unsupportedURL
    }
}

private actor ProductHTTPClient {
    func get(_ url: URL) async throws -> ProductHTTPPage {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 20
        configuration.timeoutIntervalForResource = 30
        configuration.httpShouldSetCookies = false
        let session = URLSession(configuration: configuration, delegate: ProductRedirects(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: url)
        request.setValue("text/html,text/plain;q=0.8", forHTTPHeaderField: "Accept")
        request.setValue(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15",
            forHTTPHeaderField: "User-Agent")
        let (stream, response) = try await session.bytes(for: request)
        defer { stream.task.cancel() }
        guard let response = response as? HTTPURLResponse else { throw ProductError.unsupportedContent }
        guard response.expectedContentLength <= ProductPageClient.byteLimit else { throw ProductError.pageTooLarge }
        var bytes: [UInt8] = []
        bytes.reserveCapacity(64 * 1024)
        for try await byte in stream {
            try Task.checkCancellation()
            guard bytes.count < ProductPageClient.byteLimit else { throw ProductError.pageTooLarge }
            bytes.append(byte)
        }
        return ProductHTTPPage(
            status: response.statusCode, location: response.value(forHTTPHeaderField: "Location"),
            mime: response.mimeType ?? "", data: Data(bytes))
    }
}

private final class ProductRedirects: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(
        _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}
