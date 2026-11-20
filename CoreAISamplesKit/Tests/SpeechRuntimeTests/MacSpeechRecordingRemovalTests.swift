#if os(macOS)
    import Testing

    struct MacSpeechRecordingRemovalTests {
        @Test("VRC-03 Failed recording deletion retains ownership across retries and the old recorder's destruction")
        func retriesWithoutReservingAgain() async throws {
            try await RecordingRemovalScenarios.ownership()
        }
        @Test(
            "VRC-03 Cancellation keeps priority when file deletion fails and the next valid recording first retires that file"
        )
        func cancellationPrecedence() async throws {
            try await RecordingRemovalScenarios.cancelled()
        }
        @Test("VRC-03 An already removed owned file retires on retry without blocking the next recording")
        func missingFileIsRetired() async throws {
            try await RecordingRemovalScenarios.alreadyGone()
        }
    }
#endif
