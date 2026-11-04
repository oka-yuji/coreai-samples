import FeatureStore

public enum MacModelSelection: Feature {
    public struct State: Equatable, Sendable {
        public private(set) var request = 0
        public private(set) var selecting = false
        public private(set) var unloading = false
        public init() {}

        mutating func start(unloading: Bool) {
            request += 1
            selecting = !unloading
            self.unloading = unloading
        }

        mutating func reset() {
            request += 1
            selecting = false
            unloading = false
        }

        mutating func finish(_ request: Int, unloading: Bool) {
            guard request == self.request else { return }
            if unloading { self.unloading = false } else { selecting = false }
        }
    }

    public enum Action: Equatable, Sendable {
        case choose
        case unload
        case reset
        case selected(Int)
        case unloaded(Int)
    }

    public enum Effect: Equatable, Sendable {}
    public enum EffectID: Hashable, Sendable {}

    public static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>] {
        switch action {
        case .choose: state.start(unloading: false)
        case .unload: state.start(unloading: true)
        case .reset: state.reset()
        case .selected(let request): state.finish(request, unloading: false)
        case .unloaded(let request): state.finish(request, unloading: true)
        }
        return []
    }
}
