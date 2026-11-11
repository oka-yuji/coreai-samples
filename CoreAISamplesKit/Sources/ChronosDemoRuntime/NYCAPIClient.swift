import ChronosFeature
import Foundation

public actor NYCAPIClient {
    public typealias Transport = @Sendable (URLRequest) async throws -> NYCAPIResponse
    private let transport: Transport
    private let wait: @Sendable () async throws -> Void
    private let now: @Sendable () -> Date
    private let cache: NYCWindowCache
    private var cachedCoverage: APICoverage?
    private var generation = 0

    public init(
        cacheDirectory: URL? = nil, transport: Transport? = nil, maxCachedWindows: Int = 32,
        now: @escaping @Sendable () -> Date = { Date() },
        retryDelay: @escaping @Sendable () async throws -> Void = { try await Task.sleep(for: .seconds(2)) }
    ) {
        self.transport = transport ?? { try await NYCHTTPTransport().receive($0) }
        self.now = now
        wait = retryDelay
        cache = NYCWindowCache(directory: cacheDirectory, capacity: min(32, max(1, maxCachedWindows)))
    }

    public func coverage(refresh: Bool = false) async throws -> APICoverage {
        try Task.checkCancellation()
        if !refresh, let cachedCoverage { return cachedCoverage }
        generation += 1
        let ticket = generation
        let evidence = try await fetch(NYCAPIContract.catalogURL())
        let result = try NYCCatalogValidation.read(evidence, now: evidence.acquiredAt)
        try check(ticket)
        cachedCoverage = result
        return result
    }

    public func window(start: Date, count: Int, areaIDs: [String], refresh: Bool = false) async throws -> APIWindow {
        guard start.timeIntervalSince1970.isFinite, start == NYCTime.aligned(start), (1...1024).contains(count),
            NYCAPIContract.validAreas(areaIDs)
        else { throw ChronosDemoFailure.invalidDate }
        let coverage = try await coverage(refresh: refresh)
        let ticket = generation
        if count == 1008, !coverage.allowsHistory(endingAt: start.addingTimeInterval(1008 * 1800)) {
            throw ChronosDemoFailure.unpublishedHistory
        }
        let areas = areaIDs.sorted()
        let datasets = NYCWindowValidation.datasets(start: start, count: count, coverage: coverage)
        let key = NYCAPIContract.hash(
            try NYCAPIContract.encode(
                CacheKey(
                    start: start, count: count, areas: areas, datasets: datasets, catalogHash: coverage.evidence.sha256)
            ))
        if !refresh, let saved = try? cache.read(key), saved.start == start, saved.count == count,
            saved.series.map(\.areaID) == areas, saved.datasets == datasets,
            saved.catalog.sha256 == coverage.evidence.sha256
        {
            try check(ticket, coverage: coverage)
            return saved
        }
        var metadata: [APIEvidence] = []
        var responses: [APIEvidence] = []
        var bytes = coverage.evidence.data.count
        for dataset in datasets {
            try check(ticket, coverage: coverage)
            let evidence = try await fetch(NYCAPIContract.metadataURL(dataset))
            try check(ticket, coverage: coverage)
            try NYCCatalogValidation.metadata(evidence, dataset: dataset)
            bytes += evidence.data.count
            guard bytes <= NYCAPIContract.rawLimit else { throw ChronosDemoFailure.responseTooLarge }
            metadata.append(evidence)
        }
        for chunk in try NYCWindowValidation.chunks(start: start, count: count, datasets: datasets) {
            try check(ticket, coverage: coverage)
            let url = try NYCAPIContract.queryURL(chunk.dataset, start: chunk.start, end: chunk.end, areas: areas)
            let evidence = try await fetch(url, start: chunk.start, end: chunk.end)
            try check(ticket, coverage: coverage)
            bytes += evidence.data.count
            guard bytes <= NYCAPIContract.rawLimit else { throw ChronosDemoFailure.responseTooLarge }
            responses.append(evidence)
        }
        let result = try NYCWindowValidation.make(
            start: start, count: count, areas: areas, coverage: coverage, metadata: metadata,
            responses: responses, fetchedAt: now())
        try check(ticket, coverage: coverage)
        do { try cache.save(result, key: key) } catch let error as ChronosDemoFailure { throw error } catch {
            throw ChronosDemoFailure.persistence
        }
        return result
    }

    public func validate(_ snapshot: ForecastInputSnapshot) throws -> String {
        try Task.checkCancellation()
        try NYCWindowValidation.validate(snapshot.history)
        if let actuals = snapshot.actuals { try NYCWindowValidation.validate(actuals) }
        let offset = snapshot.origin.timeIntervalSince(snapshot.history.end) / 1800
        guard offset.isFinite, (0...12).contains(offset), offset.rounded() == offset,
            try ForecastInputSnapshot(history: snapshot.history, actuals: snapshot.actuals, step: Int(offset))
                == snapshot
        else { throw ChronosDemoFailure.invalidData }
        return NYCAPIContract.hash(try NYCAPIContract.encode(snapshot))
    }

    private func check(_ ticket: Int, coverage: APICoverage? = nil) throws {
        try Task.checkCancellation()
        guard ticket == generation, coverage.map({ $0 == cachedCoverage }) ?? true else { throw CancellationError() }
    }

    private func fetch(_ url: URL, start: Date? = nil, end: Date? = nil) async throws -> APIEvidence {
        guard NYCAPIContract.validURL(url) else { throw ChronosDemoFailure.invalidData }
        var request = URLRequest(url: url)
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        for attempt in 0..<2 {
            try Task.checkCancellation()
            let response: NYCAPIResponse
            do { response = try await transport(request) } catch is CancellationError {
                throw CancellationError()
            } catch let error as URLError {
                if error.code == .cancelled { throw CancellationError() }
                if attempt == 0, [.timedOut, .networkConnectionLost].contains(error.code) {
                    try await wait()
                    continue
                }
                throw ChronosDemoFailure.network
            } catch let error as ChronosDemoFailure { throw error } catch { throw ChronosDemoFailure.network }
            try Task.checkCancellation()
            guard response.data.count <= NYCAPIContract.byteLimit else { throw ChronosDemoFailure.responseTooLarge }
            if response.status == 429 { throw ChronosDemoFailure.rateLimited }
            if (300..<400).contains(response.status) { throw ChronosDemoFailure.redirectDenied }
            if attempt == 0, [500, 502, 503, 504].contains(response.status) {
                try await wait()
                continue
            }
            guard response.status == 200 else { throw ChronosDemoFailure.network }
            return APIEvidence(
                url: url, start: start, end: end, status: response.status, acquiredAt: now(), data: response.data,
                sha256: NYCAPIContract.hash(response.data))
        }
        throw ChronosDemoFailure.network
    }

    private struct CacheKey: Encodable {
        let start: Date
        let count: Int
        let areas: [String]
        let datasets: [APIDataset]
        let catalogHash: String
        let preprocessing = APIWindow.preprocessingVersion
    }
}
