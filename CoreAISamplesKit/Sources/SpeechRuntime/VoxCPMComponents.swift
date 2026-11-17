#if os(macOS) || os(iOS)
    protocol VoxCPMComponents: Sendable {
        associatedtype States: Sendable

        func makeStates() throws -> States
        @concurrent func call(_ function: VoxCPMFunction, _ inputs: [String: VoxCPMInput], states: inout States)
            async throws -> [String: [Float]]
    }
#endif
