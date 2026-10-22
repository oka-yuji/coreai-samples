import Foundation

public struct ImageTextConditioning: Sendable {
    public let vision: [Float]
    public let deepstack: [Float]

    public init(layout: ImagePromptLayout, visionOutputs: [[Float]]?) throws {
        let slots = layout.tokens.indices.filter { layout.tokens[$0] == ImagePromptLayout.imageToken }
        if slots.isEmpty {
            guard visionOutputs == nil else { throw ImageGenerationError.invalidShape }
        } else {
            guard slots.count == 256, let visionOutputs, visionOutputs.count == 4,
                visionOutputs.allSatisfy({ $0.count == 256 * 4_096 })
            else { throw ImageGenerationError.invalidShape }
            guard visionOutputs.allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
                throw ImageGenerationError.nonFinite
            }
        }
        var vision = [Float](repeating: 0, count: 512 * 4_096)
        var deepstack = [Float](repeating: 0, count: 3 * 512 * 4_096)
        if let visionOutputs {
            for (index, slot) in slots.enumerated() {
                let source = (index * 4_096)..<((index + 1) * 4_096)
                vision.replaceSubrange((slot * 4_096)..<((slot + 1) * 4_096), with: visionOutputs[0][source])
                for tap in 0..<3 {
                    let start = (tap * 512 + slot) * 4_096
                    deepstack.replaceSubrange(start..<(start + 4_096), with: visionOutputs[tap + 1][source])
                }
            }
        }
        self.vision = vision
        self.deepstack = deepstack
    }
}

extension ImagePromptLayout {
    public func condition(fromHidden hidden: [Float]) throws -> [Float] {
        guard hidden.count == 512 * 4_096 else { throw ImageGenerationError.invalidShape }
        guard hidden.allSatisfy(\.isFinite) else { throw ImageGenerationError.nonFinite }
        let active = Array(hidden[(drop * 4_096)..<(tokens.count * 4_096)])
        return active + Array(repeating: 0, count: 512 * 4_096 - active.count)
    }
}

extension ImageDenoiserLayout {
    public func packedLatents(target: [Float], reference: [Float]?) throws -> [Float] {
        guard target.count == ImageEuler.latentCount,
            referenceCount == 0 ? reference == nil : reference?.count == ImageEuler.latentCount
        else { throw ImageGenerationError.invalidShape }
        guard target.allSatisfy(\.isFinite), reference?.allSatisfy(\.isFinite) ?? true else {
            throw ImageGenerationError.nonFinite
        }
        let values = (reference ?? []) + target
        let count = (capacity == 2_304 ? 2_048 : 1_024) * 64
        return values + Array(repeating: 0, count: count - values.count)
    }

    public func prediction(fromOutput output: [Float]) throws -> [Float] {
        guard output.count == capacity * 64 else { throw ImageGenerationError.invalidShape }
        guard output.allSatisfy(\.isFinite) else { throw ImageGenerationError.nonFinite }
        var result = [Float]()
        result.reserveCapacity(ImageEuler.latentCount)
        for index in targetIndices { result.append(contentsOf: output[(index * 64)..<((index + 1) * 64)]) }
        return result
    }
}
