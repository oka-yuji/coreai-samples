import FeatureStore
import Foundation
import ModelLibraryFeature
import ObjectFinderFeature

@MainActor
public final class ObjectFinderDemoSessions<Preview> {
    public struct Runtime {
        public let store: Store<ObjectFinderFeature>
        public let preview: Preview
        let runner: ObjectFinderRunner
    }

    private struct Entry {
        let resource: ValidatedModelResource
        let runtime: Runtime
        var session: LibraryTicket
        var permits: @MainActor () -> Bool
    }

    private let makeRuntime: @MainActor () -> (preview: Preview, service: any ObjectFinderService)
    private var entries: [DemoSelection: Entry] = [:]

    public init(makeRuntime: @escaping @MainActor () -> (preview: Preview, service: any ObjectFinderService)) {
        self.makeRuntime = makeRuntime
    }

    public func runtime(for selection: DemoSelection) -> Runtime? { entries[selection]?.runtime }

    public func bind(_ session: ModelDemoSession, permits: @escaping @MainActor () -> Bool) -> Runtime {
        let selection = session.selection
        let resource = session.resource
        if let entry = entries[selection], entry.resource == resource {
            entries[selection]?.session = session.id
            entries[selection]?.permits = permits
            return entry.runtime
        }
        let made = makeRuntime()
        let runner = ObjectFinderRunner(service: made.service) { [weak self] in
            guard let entry = self?.entries[selection], entry.resource == resource else { return false }
            return entry.permits()
        }
        let store = Store<ObjectFinderFeature>(
            initialState: .managed(folder: resource.directory, retaining: entries[selection]?.runtime.store.state),
            run: runner.run)
        let runtime = Runtime(store: store, preview: made.preview, runner: runner)
        entries[selection] = Entry(resource: resource, runtime: runtime, session: session.id, permits: permits)
        return runtime
    }

    public func stop(_ session: ModelDemoSession) async throws {
        guard let entry = entries[session.selection] else { return }
        let store = entry.runtime.store
        guard entry.session == session.id || store.state.phase == .idle else { return }
        try await entry.runtime.runner.release(store.state.session, send: store.send)
    }
}

#if os(iOS)
    extension ObjectFinderDemoSessions where Preview == ObjectFinderPreview {
        public convenience init() {
            self.init {
                let preview = ObjectFinderPreview()
                return (preview, LiveObjectFinderService(preview: preview))
            }
        }
    }
#endif
