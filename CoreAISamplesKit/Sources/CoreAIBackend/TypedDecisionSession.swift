actor TypedDecisionSession: TypedDecisionBackbone {
    private struct Shape: Equatable {
        let tokens: Int
        let capacity: Int
    }

    private var backbone: (any TypedDecisionBackbone)?
    private let reload: @Sendable () async throws -> any TypedDecisionBackbone
    private var shape: Shape?
    private var shapeChanges = 0
    private var busy = false

    init(backbone: any TypedDecisionBackbone, reload: @escaping @Sendable () async throws -> any TypedDecisionBackbone)
    {
        self.backbone = backbone
        self.reload = reload
    }

    func forward(_ ids: [Int], capacity: Int) async throws -> [Float] {
        guard !busy else { throw TypedDecisionError.busy }
        busy = true
        defer { busy = false }
        do {
            try Task.checkCancellation()
            let next = Shape(tokens: ids.count, capacity: capacity)
            if let shape, shape != next {
                if shapeChanges == 3 {
                    discard()
                } else {
                    shapeChanges += 1
                }
            }
            if backbone == nil { backbone = try await reload() }
            try Task.checkCancellation()
            let hidden = try await infer(ids, capacity: capacity)
            try Task.checkCancellation()
            shape = next
            return hidden
        } catch {
            discard()
            throw error
        }
    }

    private func infer(_ ids: [Int], capacity: Int) async throws -> [Float] {
        guard let backbone else { throw TypedDecisionError.invalidBundle("The backbone is unavailable.") }
        return try await backbone.forward(ids, capacity: capacity)
    }

    private func discard() {
        backbone = nil
        shape = nil
        shapeChanges = 0
    }
}
