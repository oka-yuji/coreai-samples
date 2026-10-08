public protocol Feature {
    associatedtype State: Equatable & Sendable
    associatedtype Action: Sendable
    associatedtype Effect: Equatable & Sendable
    associatedtype EffectID: Hashable & Sendable

    static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>]
}

public enum Command<Effect: Equatable & Sendable, ID: Hashable & Sendable>: Equatable, Sendable {

    case run(Effect, id: ID?)

    case cancel(ID)

    case cancelAll
}
