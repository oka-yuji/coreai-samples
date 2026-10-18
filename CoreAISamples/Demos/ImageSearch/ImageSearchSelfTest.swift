import EmbeddingRuntime
import FeatureStore
import Foundation
import ImageSearchFeature
import ImageSearchRuntime
import SwiftUI

#if os(macOS)
    import AppKit
#else
    import UIKit
#endif

@MainActor
enum ImageSearchSelfTest {
    private struct Query: Decodable {
        let id: String
        let text: String
        let expected: String?
        let evaluationOnly: Bool?
    }

    private struct Match: Encodable {
        let image: String
        let similarity: Double
    }

    private struct Record: Encodable {
        let id: String
        let query: String
        let seconds: Double?
        let matches: [Match]
        let displayed: [String]
        let expected: String?
        let relevantResult: Bool
        let evaluationOnly: Bool
    }

    static func run(directory: URL, folder: URL) async -> Int32 {
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            print("Image search validation requires a new output directory: \(directory.path)")
            return 1
        }
        do {
            try await perform(directory: directory, folder: folder)
            print("Image search application scenarios passed.")
            return 0
        } catch {
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try? Data("\(error)".utf8).write(to: directory.appending(path: "failure.txt"))
            print("Image search application scenarios failed: \(error)")
            return 1
        }
    }

    private static func perform(directory: URL, folder: URL) async throws {
        let precision = try EmbeddingBundle(directory: folder).precision
        let quantized = precision == "int4-int8-ptq-s128-v70"
        let harness = ImageSearchScenarioHarness()
        let store = harness.store
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        store.send(.load(folder))
        try await harness.settle()
        try require(store.state.ready, state: store.state)
        store.send(.index(ImageSearchAssembly.samples))
        try await harness.settle()
        try require(store.state.photos.count == 5, state: store.state)
        let indexingSeconds = store.state.seconds ?? 0
        guard let queryURL = Bundle.main.url(forResource: "image-search-queries", withExtension: "json") else {
            throw failure("Missing query fixtures")
        }
        let queries = try JSONDecoder().decode([Query].self, from: Data(contentsOf: queryURL))
        var records: [Record] = []
        for query in queries {
            store.send(.setQuery(query.text))
            store.send(.search)
            try await harness.settle()
            try require(store.state.hasSearched, state: store.state)
            let relevant = store.state.displayedMatches.first?.photo.url.lastPathComponent == query.expected
            if query.evaluationOnly != true && !quantized { try require(relevant, state: store.state) }
            records.append(
                Record(
                    id: query.id, query: query.text, seconds: store.state.seconds,
                    matches: store.state.matches.map {
                        Match(image: $0.photo.url.lastPathComponent, similarity: $0.similarity)
                    }, displayed: store.state.displayedMatches.map { $0.photo.url.lastPathComponent },
                    expected: query.expected, relevantResult: relevant, evaluationOnly: query.evaluationOnly == true))
            if query.id == "ja_food" {
                try snapshot(store: store, width: 1_040, height: 760, to: directory.appending(path: "gallery-ja.png"))
                try snapshot(
                    store: store, width: 390, height: 844, to: directory.appending(path: "gallery-compact-ja.png"))
            }
            if let first = store.state.displayedMatches.first?.photo {
                store.send(.select(first))
                try require(
                    store.state.selection == first && ImageSearchAssembly.original(first.url) != nil, state: store.state
                )
                store.send(.select(nil))
            } else {
                store.send(.showAllCandidates(true))
                try require(store.state.displayedMatches.count == 5, state: store.state)
                store.send(.showAllCandidates(false))
            }
        }
        store.send(.search)
        store.send(.stop)
        try await harness.settle()
        try require(store.state.canSearch && store.state.matches.isEmpty, state: store.state)
        store.send(.search)
        try await harness.settle()
        try require(store.state.matches.count == 5, state: store.state)
        store.send(.disappeared)
        try await harness.settle()
        try require(!store.state.ready && store.state.photos.isEmpty && !store.state.isBusy, state: store.state)
        try await ImageSearchVectorCheck.run(folder: folder, output: directory.appending(path: "vectors.json"))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(records).write(to: directory.appending(path: "searches.json"), options: .atomic)
        let summary: [String: Any] = [
            "passed": true, "images": 5, "queries": records.count,
            "precision": precision,
            "vectorReference": quantized ? "Matched PTQ" : "Original FP32",
            "originalExamplesPassed": records.filter { !$0.evaluationOnly }.allSatisfy(\.relevantResult),
            "indexingSeconds": indexingSeconds, "queuedStopAndRetry": true, "originalRead": true,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "relevanceCorrect": records.filter(\.relevantResult).count,
            "heldOutQueries": records.filter(\.evaluationOnly).count,
            "heldOutCorrect": records.filter { $0.evaluationOnly && $0.relevantResult }.count,
            "minimumSimilarity": ImageSearchState.minimumSimilarity,
            "relevanceScope": "Fixed examples; reported separately from numerical and interaction checks",
        ]
        try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appending(path: "summary.json"), options: .atomic)
    }

    private static func snapshot(store: Store<ImageSearchFeature>, width: CGFloat, height: CGFloat, to url: URL) throws
    {
        let snapshotStore = Store<ImageSearchFeature>(initialState: store.state) { _, _ in }
        let view = ImageSearchView(store: snapshotStore).environment(\.locale, Locale(identifier: "ja"))
            .frame(width: width, height: height).background(.background)
        #if os(macOS)
            _ = NSApplication.shared
            let host = NSHostingView(rootView: view)
            let window = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: width, height: height),
                styleMask: .borderless, backing: .buffered, defer: false)
            window.contentView = host
            host.frame = NSRect(x: 0, y: 0, width: width, height: height)
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)
            else { throw failure("Could not render the gallery") }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            guard let png = bitmap.representation(using: .png, properties: [:]) else {
                throw failure("Could not encode the gallery")
            }
        #else
            let controller = UIHostingController(rootView: view)
            controller.view.frame = CGRect(x: 0, y: 0, width: width, height: height)
            controller.view.setNeedsLayout()
            controller.view.layoutIfNeeded()
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1
            let renderer = UIGraphicsImageRenderer(size: controller.view.bounds.size, format: format)
            let image = renderer.image { context in controller.view.layer.render(in: context.cgContext) }
            guard let png = image.pngData() else { throw failure("Could not render the gallery") }
        #endif
        try png.write(to: url)
    }

    private static func require(_ condition: Bool, state: ImageSearchState) throws {
        if !condition {
            throw failure(state.failure?.detail ?? "Application scenario did not reach the expected state")
        }
    }

    private static func failure(_ message: String) -> NSError {
        NSError(domain: "ImageSearchScenario", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}

@MainActor
private final class ImageSearchScenarioHarness {
    let store: Store<ImageSearchFeature>
    private let events: AsyncStream<Void>

    init() {
        let signals = AsyncStream<Void>.makeStream()
        events = signals.stream
        let runner = ImageSearchRunner(service: LiveImageSearchService())
        store = Store(initialState: ImageSearchState()) { effect, send in
            await runner.run(effect, send: send)
            signals.continuation.yield(())
        }
    }

    func settle() async throws {
        var iterator = events.makeAsyncIterator()
        while store.state.isBusy {
            try Task.checkCancellation()
            guard await iterator.next() != nil else { throw CancellationError() }
        }
    }
}
