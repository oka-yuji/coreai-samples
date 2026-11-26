import Foundation

public enum PreparedCopyError: Error, Equatable, LocalizedError {
    case preparing, recoveryFailed, reloadRequired, inUse

    public var errorDescription: String? {
        switch self {
        case .preparing: "The prepared copy is being used or repaired. Try loading the model again."
        case .recoveryFailed:
            "The prepared copy could not be repaired. Clear the prepared copy and load the model again."
        case .reloadRequired: "Load the model again to repair its prepared copy."
        case .inUse: "The prepared copy is in use."
        }
    }
}
