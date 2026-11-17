enum VoxCPMDescriptorDefect: CaseIterable, Sendable {
    case inputType, outputShape, missingState, stateShape, extraInput, renamedOutput, reorderedInputs
    case extraFunction, missingFunction
}
