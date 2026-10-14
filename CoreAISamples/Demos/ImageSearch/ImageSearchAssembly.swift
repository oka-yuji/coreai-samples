import FeatureStore
import Foundation
import ImageIO
import ImageSearchFeature
import ImageSearchRuntime
import SwiftUI

@MainActor
enum ImageSearchAssembly {
    static func makeStore() -> Store<ImageSearchFeature> {
        let prepared = URL.documentsDirectory.appending(path: "embeddinggemma2")
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
