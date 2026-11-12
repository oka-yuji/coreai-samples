import ChronosFeature
import Foundation
import Synchronization

public struct NYCAPIResponse: Sendable {
    public let data: Data
    public let status: Int

    public init(data: Data, status: Int) {
        self.data = data
        self.status = status
    }
}

final class NYCHTTPTransport: NSObject, URLSessionDataDelegate, Sendable {
    private struct State {
        var continuation: CheckedContinuation<NYCAPIResponse, any Error>?
        var task: URLSessionDataTask?
        var completed = false
        var data = Data()
        var status: Int?
    }

    private let state = Mutex(State())
    private let limit: Int
    private let start: @Sendable (URLSessionDataTask) -> Void

    init(
        limit: Int = NYCAPIContract.byteLimit, start: @escaping @Sendable (URLSessionDataTask) -> Void = { $0.resume() }
    ) {
        self.limit = limit
        self.start = start
    }

    func receive(_ request: URLRequest) async throws -> NYCAPIResponse {
        guard let url = request.url, NYCAPIContract.validURL(url) else { throw ChronosDemoFailure.invalidData }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCredentialStorage = nil
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 120
        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                let task = state.withLock { value -> URLSessionDataTask? in
                    guard !value.completed else {
                        continuation.resume(throwing: CancellationError())
                        return nil
                    }
                    value.continuation = continuation
                    let task = session.dataTask(with: request)
                    value.task = task
                    return task
                }
                if let task { start(task) }
            }
        } onCancel: {
            self.finish(.failure(CancellationError()))
        }
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        completionHandler(nil)
        finish(.failure(ChronosDemoFailure.redirectDenied))
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping @Sendable (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        completionHandler(
            challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust
                ? .performDefaultHandling : .cancelAuthenticationChallenge, nil)
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        guard let response = response as? HTTPURLResponse, response.url == dataTask.originalRequest?.url else {
            completionHandler(.cancel)
            finish(.failure(ChronosDemoFailure.invalidData))
            return
        }
        guard response.expectedContentLength <= Int64(limit) else {
            completionHandler(.cancel)
            finish(.failure(ChronosDemoFailure.responseTooLarge))
            return
        }
        state.withLock { $0.status = response.statusCode }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        let overflow = state.withLock { value -> Bool in
            guard !value.completed else { return false }
            guard data.count <= limit - value.data.count else { return true }
            value.data.append(data)
            return false
        }
        if overflow { finish(.failure(ChronosDemoFailure.responseTooLarge)) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        if let error {
            finish(.failure(error))
            return
        }
        let result: Result<NYCAPIResponse, any Error> = state.withLock { value in
            guard let status = value.status else { return .failure(ChronosDemoFailure.network) }
            return .success(NYCAPIResponse(data: value.data, status: status))
        }
        finish(result)
    }

    private func finish(_ result: Result<NYCAPIResponse, any Error>) {
        let pending = state.withLock {
            value -> (CheckedContinuation<NYCAPIResponse, any Error>?, URLSessionDataTask?) in
            guard !value.completed else { return (nil, nil) }
            value.completed = true
            let pending = (value.continuation, value.task)
            value.continuation = nil
            value.task = nil
            value.data = Data()
            return pending
        }
        pending.1?.cancel()
        pending.0?.resume(with: result)
    }
}
