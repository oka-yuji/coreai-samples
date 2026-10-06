/// A feature is a pure state machine. `reduce` updates the state for one action and
/// returns the effects to run as data. Running the effects is the store's job.
public protocol Feature {
    associatedtype State: Equatable & Sendable
    associatedtype Action: Sendable
    associatedtype Effect: Equatable & Sendable
    associatedtype EffectID: Hashable & Sendable

    static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>]
}

/// An effect requested by `Feature.reduce`. The store handles commands in the order they are returned.
///
/// Cancellation is cooperative (SE-0304) and does not mean the runner has stopped. For a resource that must
/// finish stopping before the next use starts (a camera, a microphone), model a "stop" effect and a "stopping"
/// state instead of cancelling.
public enum Command<Effect: Equatable & Sendable, ID: Hashable & Sendable>: Equatable, Sendable {
    /// Runs `effect`. With an id, first cancels the effect running under the same id.
    case run(Effect, id: ID?)
    /// Cancels the effect running under the id. Results that arrive after the cancellation are dropped.
    case cancel(ID)
    /// Cancels every running effect. Results that arrive after the cancellation are dropped.
    /// Return it from the action that ends the feature (such as closing the screen).
    case cancelAll
}
