import ChronosFeature
import FeatureStore
import ModelLibraryRuntime
import Observation

@MainActor
final class ChronosPublicationAttempt {
    @Observable
    fileprivate final class Lifetime { var active = true }

    let permit = ChronosPublicationPermit()
    private let lifetime = Lifetime()
    private var permission: DemoExecutionPermission?
    private var journal: Store<ChronosJournalFeature>?

    init(
        permission: DemoExecutionPermission, journal: Store<ChronosJournalFeature>?,
        effect: ChronosJournalFeature.Effect
    ) throws {
        self.permission = permission
        self.journal = journal
        let permit = permit
        let allowed = withObservationTracking(options: [.willSet, .deinit]) {
            lifetime.active && permission.isPermitted()
                && (journal.map { ChronosJournalFeature.permits(effect, in: $0.state) } ?? true)
        } onChange: { event in
            permit.invalidate()
            event.cancel()
        }
        guard allowed, !Task.isCancelled else {
            finish()
            throw CancellationError()
        }
    }

    func finish() {
        lifetime.active = false
        permission = nil
        journal = nil
    }
}
