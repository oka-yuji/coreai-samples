#if os(macOS)
    import Dispatch

    struct MacSpeechRecordingDeadline: Sendable {
        static let live = MacSpeechRecordingDeadline(
            now: { ContinuousClock.now },
            schedule: { remaining, fire in
                let source = DispatchSource.makeTimerSource(queue: .global())
                source.setEventHandler { @Sendable in fire(ContinuousClock.now) }
                source.schedule(deadline: .now() + dispatchInterval(remaining), repeating: .never)
                source.activate()
                return { source.cancel() }
            })

        let now: @Sendable () -> ContinuousClock.Instant
        let schedule: @Sendable (Duration, @escaping @Sendable (ContinuousClock.Instant) -> Void) -> () -> Void

        private static func dispatchInterval(_ remaining: Duration) -> DispatchTimeInterval {
            let (seconds, attoseconds) = max(.zero, remaining).components
            guard seconds < Int64.max / 1_000_000_000 else { return .nanoseconds(Int.max) }
            return .nanoseconds(Int(seconds * 1_000_000_000 + attoseconds / 1_000_000_000))
        }
    }
#endif
