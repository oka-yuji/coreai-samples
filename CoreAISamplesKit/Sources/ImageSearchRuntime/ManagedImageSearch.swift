import FeatureStore
import Foundation
import ImageSearchFeature

@MainActor
public final class ManagedImageSearch {
    private struct Bound {
        let token: Int
        let folder: URL
        let identity: String
        let store: Store<ImageSearchFeature>
        let runner: ImageSearchRunner
    }

    private let makeService: @MainActor () -> any ImageSearchService
    private var bound: Bound?
    private var retired: [Bound] = []
    private var bindings = 0
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
        bindings += 1
        let token = bindings
        let runner = ImageSearchRunner(service: makeService()) { [weak self] in
            guard let self, bound?.token == token else { return false }
            return permission()
        }
        let store = Store<ImageSearchFeature>(
            initialState: ImageSearchState(managed: folder, identity: identity, retaining: bound?.store.state),
            run: runner.run)
        if let bound { retired.append(bound) }
        bound = Bound(token: token, folder: folder, identity: identity, store: store, runner: runner)
        return store
    }

    public func release() async throws {
        var failure: (any Error)?
        for target in (bound.map { [$0] } ?? []) + retired {
            target.store.send(.closed)
            do {
                try await target.runner.release(target.store.state.session)
                retired.removeAll { $0.token == target.token }
            } catch {
                failure = failure ?? error
            }
        }
        if let failure { throw failure }
    }
}
