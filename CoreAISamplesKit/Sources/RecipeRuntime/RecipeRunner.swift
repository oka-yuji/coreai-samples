import Foundation
import RecipeCore
import RecipeFeature

@MainActor
public final class RecipeRunner {
    private let service: any RecipeService
    private let saveModelPath: @MainActor (String) -> Void

    public init(service: any RecipeService, saveModelPath: @escaping @MainActor (String) -> Void = { _ in }) {
        self.service = service
        self.saveModelPath = saveModelPath
    }

    public func run(_ effect: RecipeEffect, send: (RecipeAction) -> Void) async {
        let session: Int
        switch effect {
        case .inspect(let id, _), .fetch(let id, _), .loadModel(let id, _), .check(let id, _, _), .waitForIdle(let id):
            session = id
        }
        do {
            switch effect {
            case .inspect(let id, let url):
                let accessing = url.startAccessingSecurityScopedResource()
                defer { if accessing { url.stopAccessingSecurityScopedResource() } }
                let photo = try await service.inspect(url)
                try Task.checkCancellation()
                send(.photoLoaded(id, photo))
            case .fetch(let id, let url):
                let recipe = try await service.fetch(url)
                try Task.checkCancellation()
                send(.recipeLoaded(id, recipe))
            case .loadModel(let id, let path):
                let name = try await service.loadModel(path)
                try Task.checkCancellation()
                saveModelPath(path)
                send(.modelLoaded(id, name))
            case .check(let id, let revision, let input):
                let report = try await service.check(input)
                try Task.checkCancellation()
                send(.checked(id, revision, report))
            case .waitForIdle(let id):
                await service.waitForIdle()
                send(.stopped(id))
            }
        } catch {
            if !Task.isCancelled && !(error is CancellationError) {
                send(.failed(session, error.localizedDescription))
            }
        }
    }
}
