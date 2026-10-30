public struct TeleOCRModelNotice: Equatable, Sendable {
    public let message: String
    public let detail: String?

    public init(_ failure: ModelLibraryFailure) {
        switch failure {
        case .operation("insufficientSpace"):
            message = "There is not enough disk space to import this model. Free some space and try again."
            detail = nil
        case .operation("ledgerCorrupted"):
            message = "The model library record is damaged. Model changes are unavailable."
            detail = nil
        case .operation("transfer in progress"):
            message = "Another window is transferring this model. Wait for it to finish."
            detail = nil
        case .operation(let reason):
            message = "The model operation could not be completed."
            detail = reason
        case .modelInUse:
            message = "Close the OCR demo before replacing or removing its model."
            detail = nil
        default:
            message = "The model is not available. Check its folder and try importing again."
            detail = nil
        }
    }
}
