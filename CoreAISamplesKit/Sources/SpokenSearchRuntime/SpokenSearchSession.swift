import FeatureStore
import SpokenSearchFeature

@MainActor
public final class SpokenSearchSession {
    public let store: Store<SpokenSearchFeature>
    private let runner: SpokenSearchRunner

    public init(isPermitted: @escaping @MainActor () -> Bool = { true }) {
        let runner = SpokenSearchRunner(isPermitted: isPermitted)
        self.runner = runner
        store = Store(initialState: SpokenSearchState(), run: runner.run)
    }

    public func suspend() {
        store.send(.disappeared)
        runner.stopPlayback()
    }

    public func close() async {
        suspend()
        await runner.close()
    }
}
