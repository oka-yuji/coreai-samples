import FeatureStore
import Foundation
import ImageIO
import ImageSearchFeature
import ImageSearchRuntime
import SwiftUI

@MainActor
enum ImageSearchAssembly {
    static var preparedFolderName: String {
        LaunchArguments.value("--embedding-profile") == "int4_int8" ? "embeddinggemma2-int4-int8" : "embeddinggemma2"
    }

    static var preparedModels: [URL] {
        ["embeddinggemma2", "embeddinggemma2-int4-int8"]
            .map { URL.documentsDirectory.appending(path: $0) }
            .filter { FileManager.default.fileExists(atPath: $0.appending(path: "embedding-manifest.json").path) }
    }

    struct Query: Decodable, Identifiable {
        let id: String
        let text: String
    }

    static var queries: [Query] {
        guard let url = Bundle.main.url(forResource: "image-search-queries", withExtension: "json"),
            let data = try? Data(contentsOf: url), let queries = try? JSONDecoder().decode([Query].self, from: data)
        else { return [] }
        return queries
    }

    static func makeStore() -> Store<ImageSearchFeature> {
        let prepared = URL.documentsDirectory.appending(path: preparedFolderName)
        let folder =
            LaunchArguments.value("--embedding-bundle").map { URL(filePath: $0) }
            ?? (FileManager.default.fileExists(atPath: prepared.appending(path: "embedding-manifest.json").path)
                ? prepared : nil)
        return Store(
            initialState: ImageSearchState(modelFolder: folder),
            run: ImageSearchRunner(service: LiveImageSearchService()).run)
    }

    static var samples: [URL] {
        ["01-oyakodon", "02-ginger-pork", "03-omelette", "04-gratin", "05-caprese"]
            .compactMap { Bundle.main.url(forResource: $0, withExtension: "png") }
    }

    static func image(data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    static func original(_ url: URL) -> CGImage? {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateThumbnailAtIndex(
            source, 0,
            [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 32_768,
                kCGImageSourceShouldCacheImmediately: true,
            ] as CFDictionary)
    }
}
