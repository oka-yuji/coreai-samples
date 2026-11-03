import FeatureStore
import Foundation

public struct ModelUsage: Equatable, Sendable {
    public internal(set) var resource: ValidatedModelResource?
    var users = 0
    var held: [LibraryTicket] = []
    var pending: [LibraryTicket] = []
    var refused: ModelArtifactIdentity?

    public var isActive: Bool { !held.isEmpty || !pending.isEmpty }
}

extension ModelLibraryFeature {
    static func endUsage(_ state: inout State, _ modelID: String) -> [Operation] {
        guard var usage = state.usages[modelID], usage.users > 0 else { return [] }
        usage.users -= 1
        var operations: [Operation] = []
        if usage.held.count > usage.users, let ticket = usage.held.popLast(), let resource = usage.resource {
            operations.append(.run(.releaseUsage(resource.identity, ticket), id: nil))
        }
        keep(&state, usage, for: modelID)
        return operations
    }

    static func acquired(
        _ state: inout State, _ modelID: String, _ ticket: LibraryTicket, _ resource: ValidatedModelResource?
    ) -> [Operation] {
        guard var usage = state.usages[modelID], let index = usage.pending.firstIndex(of: ticket) else { return [] }
        usage.pending.remove(at: index)
        let installed = state.selectedArtifact(for: modelID).flatMap { state.installations[$0.id] }
        var operations: [Operation] = []
        if let resource {
            if usage.held.count < usage.users, installed?.phase == .ready, installed?.resource == resource,
                usage.resource == nil || usage.resource == resource
            {
                usage.held.append(ticket)
                usage.resource = resource
            } else {
                operations.append(.run(.releaseUsage(resource.identity, ticket), id: nil))
            }
        } else if installed?.phase == .ready {
            usage.refused = installed?.resource?.identity
        }
        keep(&state, usage, for: modelID)
        return operations
    }

    static func keep(_ state: inout State, _ usage: ModelUsage, for modelID: String) {
        var usage = usage
        if usage.held.isEmpty { usage.resource = nil }
        state.usages[modelID] = usage.users == 0 && !usage.isActive ? nil : usage
    }

    static func serveUsages(_ state: inout State) -> [Operation] {
        var operations: [Operation] = []
        for modelID in state.usages.keys.sorted() {
            guard var usage = state.usages[modelID] else { continue }
            let installed = state.selectedArtifact(for: modelID).flatMap { state.installations[$0.id] }
            guard installed?.phase == .ready, let resource = installed?.resource else {
                state.usages[modelID]?.refused = nil
                continue
            }
            guard usage.refused != resource.identity, usage.resource == nil || usage.resource == resource else {
                continue
            }
            while usage.held.count + usage.pending.count < usage.users {
                let ticket = state.allocateID()
                usage.pending.append(ticket)
                operations.append(.run(.acquireUsage(resource.identity, ticket), id: nil))
            }
            state.usages[modelID] = usage
        }
        return operations
    }
}
