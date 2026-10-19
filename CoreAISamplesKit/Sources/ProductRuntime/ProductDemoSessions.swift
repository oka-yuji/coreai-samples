import FeatureStore
import Foundation
import ModelLibraryFeature
import ProductFeature

@MainActor
public final class ProductDemoSessions {
    private struct Entry {
        let identity: ModelArtifactIdentity
        let store: Store<ProductFeature>
        let runner: ProductRunner
        var permits: @MainActor () -> Bool
    }

    private let makeService: @MainActor () -> any ProductService
    private var entries: [DemoSelection: Entry] = [:]

    public init(makeService: @escaping @MainActor () -> any ProductService = { LiveProductService() }) {
        self.makeService = makeService
    }

    public func store(for selection: DemoSelection) -> Store<ProductFeature>? { entries[selection]?.store }

    public func bind(_ session: ModelDemoSession, permits: @escaping @MainActor () -> Bool) -> Store<ProductFeature> {
        let selection = session.selection
        let identity = session.identity
        if let entry = entries[selection], entry.identity == identity {
            entries[selection]?.permits = permits
            return entry.store
        }
        let runner = ProductRunner(service: makeService()) { [weak self] in
            guard let entry = self?.entries[selection], entry.identity == identity else { return false }
            return entry.permits()
        }
        let state = ProductState(
            folder: session.resource.directory,
            resourceIdentity: "\(identity.modelID)/\(identity.artifactID)@\(identity.revision)",
            keeping: entries[selection]?.store.state)
        let store = Store<ProductFeature>(initialState: state, run: runner.run)
        entries[selection] = Entry(identity: identity, store: store, runner: runner, permits: permits)
        return store
    }

    public func stop(_ session: ModelDemoSession) async throws {
        guard let entry = entries[session.selection], entry.identity == session.identity else { return }
        entry.store.send(.closed)
        try await entry.runner.release()
    }
}
