import FeatureStore

public enum ModelRuntimeUses: Feature {
    public struct State: Equatable, Sendable {
        public private(set) var counts: [String: Int] = [:]
        public init() {}

        public mutating func change(_ model: String, by delta: Int) {
            let count = max(0, counts[model, default: 0] + delta)
            counts[model] = count == 0 ? nil : count
        }
    }

    public enum Action: Equatable, Sendable {
        case acquired(String)
        case released(String)
    }

    public enum Effect: Equatable, Sendable {}
    public enum EffectID: Hashable, Sendable {}

    public static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>] {
        switch action {
        case .acquired(let model): state.change(model, by: 1)
        case .released(let model): state.change(model, by: -1)
        }
        return []
    }
}
