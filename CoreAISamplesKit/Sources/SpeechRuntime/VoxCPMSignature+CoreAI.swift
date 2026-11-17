#if os(macOS) || os(iOS)
    import CoreAI

    extension VoxCPMSignature {
        init(_ descriptor: InferenceFunctionDescriptor) {
            func tensors(_ names: [String], _ lookup: (String) -> InferenceValue.Descriptor?) -> [Tensor] {
                names.map { name in
                    guard case .ndArray(let array) = lookup(name), array.interleaveLayout == nil else {
                        return Tensor(name: name, type: "unsupported", shape: [])
                    }
                    return Tensor(name: name, type: Self.type(array.scalarType), shape: array.shape)
                }
            }
            self.init(
                inputs: tensors(descriptor.inputNames, descriptor.inputDescriptor(of:)),
                outputs: tensors(descriptor.outputNames, descriptor.outputDescriptor(of:)),
                states: tensors(descriptor.stateNames, descriptor.stateDescriptor(of:)))
        }

        private static func type(_ scalarType: NDArray.ScalarType) -> String {
            switch scalarType {
            case .float32: "float32"
            case .float16: "float16"
            case .int32: "int32"
            default: "\(scalarType)"
            }
        }
    }
#endif
