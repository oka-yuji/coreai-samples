import Foundation

public enum OCRError: Error, Equatable, Sendable {
    case invalidImage
    case invalidShape
    case nonFiniteLogits
    case generationFinished
}
