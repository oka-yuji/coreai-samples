import Foundation
import RecipeCore

private final class RecipeRedirectPolicy: NSObject, URLSessionTaskDelegate {
    func urlSession(
        _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

struct RecipeHTTPPage: Sendable {
    let status: Int
    let location: String?
    let data: Data
}

private actor RecipeHTTPClient {
    private let session: URLSession

    init() {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 20
        configuration.timeoutIntervalForResource = 30
        configuration.httpShouldSetCookies = false
        session = URLSession(configuration: configuration, delegate: RecipeRedirectPolicy(), delegateQueue: nil)
    }

    deinit {
        session.invalidateAndCancel()
    }

    func get(_ url: URL) async throws -> RecipeHTTPPage {
        var request = URLRequest(url: url)
        request.setValue("text/html", forHTTPHeaderField: "Accept")
        let (bytes, response) = try await session.bytes(for: request)
        guard let response = response as? HTTPURLResponse else {
            throw RecipeError.invalid("The recipe server did not return an HTTP response.")
        }
        guard response.expectedContentLength <= 2_000_000 else {
            throw RecipeError.invalid("The recipe page is too large.")
        }
        var data = Data()
        for try await byte in bytes {
            try Task.checkCancellation()
            data.append(byte)
            guard data.count <= 2_000_000 else { throw RecipeError.invalid("The recipe page is too large.") }
        }
        return RecipeHTTPPage(
            status: response.statusCode, location: response.value(forHTTPHeaderField: "Location"), data: data)
    }
}

public actor RecipeFetcher {
    private let get: @Sendable (URL) async throws -> RecipeHTTPPage

    public init() {
        let client = RecipeHTTPClient()
        get = { url in try await client.get(url) }
    }

    init(get: @escaping @Sendable (URL) async throws -> RecipeHTTPPage) {
        self.get = get
    }

    public func fetch(_ source: String) async throws -> Recipe {
        var url = try RecipeURL.validate(source)
        for _ in 0..<4 {
            try Task.checkCancellation()
            let response = try await get(url)
            try Task.checkCancellation()
            guard response.data.count <= 2_000_000 else { throw RecipeError.invalid("The recipe page is too large.") }
            if (300..<400).contains(response.status) {
                guard let location = response.location,
                    let destination = URL(string: location, relativeTo: url)?.absoluteURL
                else { throw RecipeError.invalid("The recipe redirect is missing its destination.") }
                url = try RecipeURL.validate(destination.absoluteString)
                continue
            }
            guard response.status == 200 else {
                throw RecipeError.invalid("The recipe server returned HTTP \(response.status).")
            }
            guard let html = String(data: response.data, encoding: .utf8) else {
                throw RecipeError.invalid("The recipe page is not UTF-8 text.")
            }
            return try RecipeParser.parse(html: html, source: url.absoluteString)
        }
        throw RecipeError.invalid("The recipe page redirected too many times.")
    }
}
