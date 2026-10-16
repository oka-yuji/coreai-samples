import Foundation
import TruckInspectionCore
import TruckInspectionFeature

@MainActor
public protocol TruckInspectionModelService {
    func prepare(folder: URL) async throws -> InspectionModelInfo
    func classify(text: String, item: InspectionItem, catalog: InspectionCatalog) async throws -> InspectionCandidate
    func waitForIdle() async
    func unload()
}

@MainActor
public struct TruckInspectionRunner {
    private let model: any TruckInspectionModelService

    public init(model: any TruckInspectionModelService) { self.model = model }

    public func run(_ effect: InspectionEffect, send: (InspectionAction) -> Void) async {
        let id: Int
        switch effect {
        case .prepare(let value, _), .classify(let value, _, _, _), .stop(let value, _): id = value
        }
        do {
            switch effect {
            case .prepare(let id, let folder):
                let info = try await model.prepare(folder: folder)
                try Task.checkCancellation()
                send(.modelLoaded(id, info))
            case .classify(let id, let text, let item, let catalog):
                let clock = ContinuousClock()
                let start = clock.now
                let candidate = try await model.classify(text: text, item: item, catalog: catalog)
                try Task.checkCancellation()
                send(.classified(id, candidate, (clock.now - start) / .milliseconds(1)))
            case .stop(let id, let unload):
                await model.waitForIdle()
                try Task.checkCancellation()
                if unload { model.unload() }
                send(.stopped(id, unloaded: unload))
            }
        } catch {
            if !Task.isCancelled && !(error is CancellationError) {
                send(.failed(id, String(describing: error)))
            }
        }
    }
}
