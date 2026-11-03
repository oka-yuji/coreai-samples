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
        Store(
            initialState: VideoSearchState(modelFolder: ImageSearchAssembly.initialFolder),
            run: VideoSearchRunner(service: LiveVideoSearchService()).run)
    }
}
