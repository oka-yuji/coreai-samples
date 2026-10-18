import FeatureStore
import Foundation
import ImageSearchFeature

@MainActor
public final class ManagedImageSearch {
    private struct Bound {
        let folder: URL
        let identity: String
        let store: Store<ImageSearchFeature>
        let runner: ImageSearchRunner
    }

    private let makeService: @MainActor () -> any ImageSearchService
    private var bound: Bound?
    private var permission: @MainActor () -> Bool = { false }

    public init(makeService: @escaping @MainActor () -> any ImageSearchService) {
        self.makeService = makeService
    }

    public var state: ImageSearchState? { bound?.store.state }

    public func store(
        folder: URL, identity: String, isPermitted: @escaping @MainActor () -> Bool
    ) -> Store<ImageSearchFeature> {
        permission = isPermitted
        if let bound, bound.folder == folder, bound.identity == identity { return bound.store }
        let runner = ImageSearchRunner(service: makeService()) { [weak self] in
            guard let self, bound?.folder == folder, bound?.identity == identity else { return false }
            return permission()
        }
        let store = Store<ImageSearchFeature>(
            initialState: ImageSearchState(managed: folder, identity: identity, retaining: bound?.store.state),
            run: runner.run)
        bound = Bound(folder: folder, identity: identity, store: store, runner: runner)
        return store
    }

    public func release() async throws {
        guard let bound else { return }
        bound.store.send(.closed)
        try await bound.runner.release()
    }
}
