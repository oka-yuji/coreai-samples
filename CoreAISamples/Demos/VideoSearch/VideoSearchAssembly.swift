import FeatureStore
import Foundation
import VideoSearchFeature
import VideoSearchRuntime

@MainActor
enum VideoSearchAssembly {
    static var queries: [ImageSearchAssembly.Query] {
        guard let url = Bundle.main.url(forResource: "video-search-queries", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let queries = try? JSONDecoder().decode([ImageSearchAssembly.Query].self, from: data)
        else { return [] }
        return queries
    }

    static func makeStore() -> Store<VideoSearchFeature> {
        let prepared = URL.documentsDirectory.appending(path: ImageSearchAssembly.preparedFolderName)
        let folder =
            LaunchArguments.value("--embedding-bundle").map { URL(filePath: $0) }
            ?? (FileManager.default.fileExists(atPath: prepared.appending(path: "embedding-manifest.json").path)
                ? prepared : nil)
        return Store(
            initialState: VideoSearchState(modelFolder: folder),
            run: VideoSearchRunner(service: LiveVideoSearchService()).run)
    }
}
