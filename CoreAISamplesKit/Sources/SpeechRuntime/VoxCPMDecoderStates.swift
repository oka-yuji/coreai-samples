#if os(macOS) || os(iOS)
    import CoreAI

    struct VoxCPMDecoderStates: Sendable {
        var state00: NDArray
        var state01: NDArray
        var state02: NDArray
        var state03: NDArray
        var state04: NDArray
        var state05: NDArray
        var state06: NDArray
        var state07: NDArray
        var state08: NDArray
        var state09: NDArray
        var state10: NDArray
        var state11: NDArray
        var state12: NDArray
        var state13: NDArray
        var state14: NDArray
        var state15: NDArray
        var state16: NDArray
        var state17: NDArray
        var state18: NDArray
        var state19: NDArray
        var state20: NDArray
        var state21: NDArray
        var state22: NDArray
        var state23: NDArray
        var state24: NDArray
        var state25: NDArray

        init(_ make: (String) throws -> NDArray) rethrows {
            let names = VoxCPMContract.decoderStateNames
            state00 = try make(names[0])
            state01 = try make(names[1])
            state02 = try make(names[2])
            state03 = try make(names[3])
            state04 = try make(names[4])
            state05 = try make(names[5])
            state06 = try make(names[6])
            state07 = try make(names[7])
            state08 = try make(names[8])
            state09 = try make(names[9])
            state10 = try make(names[10])
            state11 = try make(names[11])
            state12 = try make(names[12])
            state13 = try make(names[13])
            state14 = try make(names[14])
            state15 = try make(names[15])
            state16 = try make(names[16])
            state17 = try make(names[17])
            state18 = try make(names[18])
            state19 = try make(names[19])
            state20 = try make(names[20])
            state21 = try make(names[21])
            state22 = try make(names[22])
            state23 = try make(names[23])
            state24 = try make(names[24])
            state25 = try make(names[25])
        }

        @concurrent
        static func run(_ function: InferenceFunction, _ inputs: [String: NDArray], _ states: inout Self)
            async throws -> InferenceFunction.Outputs
        {
            let names = VoxCPMContract.decoderStateNames
            var views = InferenceFunction.MutableViews()
            views.insert(&states.state00, for: names[0])
            views.insert(&states.state01, for: names[1])
            views.insert(&states.state02, for: names[2])
            views.insert(&states.state03, for: names[3])
            views.insert(&states.state04, for: names[4])
            views.insert(&states.state05, for: names[5])
            views.insert(&states.state06, for: names[6])
            views.insert(&states.state07, for: names[7])
            views.insert(&states.state08, for: names[8])
            views.insert(&states.state09, for: names[9])
            views.insert(&states.state10, for: names[10])
            views.insert(&states.state11, for: names[11])
            views.insert(&states.state12, for: names[12])
            views.insert(&states.state13, for: names[13])
            views.insert(&states.state14, for: names[14])
            views.insert(&states.state15, for: names[15])
            views.insert(&states.state16, for: names[16])
            views.insert(&states.state17, for: names[17])
            views.insert(&states.state18, for: names[18])
            views.insert(&states.state19, for: names[19])
            views.insert(&states.state20, for: names[20])
            views.insert(&states.state21, for: names[21])
            views.insert(&states.state22, for: names[22])
            views.insert(&states.state23, for: names[23])
            views.insert(&states.state24, for: names[24])
            views.insert(&states.state25, for: names[25])
            return try await function.run(inputs: inputs, states: views)
        }
    }
#endif
