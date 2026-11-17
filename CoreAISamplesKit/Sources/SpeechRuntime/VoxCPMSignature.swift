#if os(macOS) || os(iOS)
    import SpeechAssets

    struct VoxCPMSignature: Equatable, Sendable {
        struct Tensor: Equatable, Sendable {
            let name: String
            let type: String
            let shape: [Int]
        }

        let inputs: [Tensor]
        let outputs: [Tensor]
        let states: [Tensor]

        init(inputs: [Tensor], outputs: [Tensor], states: [Tensor]) {
            self.inputs = inputs
            self.outputs = outputs
            self.states = states
        }

        init(_ function: VoxCPMBundleManifest.Function) {
            func tensors(_ values: [VoxCPMBundleManifest.Tensor]) -> [Tensor] {
                values.map { Tensor(name: $0.name, type: $0.dtype, shape: $0.shape) }
            }
            self.init(
                inputs: tensors(function.inputs), outputs: tensors(function.outputs), states: tensors(function.states))
        }
    }
#endif
