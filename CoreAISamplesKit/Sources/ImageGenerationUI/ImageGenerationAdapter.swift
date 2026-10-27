import Foundation
import ImageGenerationFeature
import ImageGenerationRuntime
import ModelLibraryFeature
import ModelLibraryRuntime
import SwiftUI

@MainActor
public final class ImageGenerationAdapter: ModelDemoAdapter {
    public let demoID = "image-generation"
    private let makeService: ManagedImageGeneration.Factory
    private var generations: [DemoSelection: ManagedImageGeneration] = [:]

    public init(makeService: @escaping ManagedImageGeneration.Factory) { self.makeService = makeService }

    public func preview(for selection: DemoSelection) -> AnyView {
        let state = ImageGenerationState(
            owner: UUID(), availability: .needsModel, retaining: generations[selection]?.state)
        return AnyView(ImageGenerationContent(state: state, send: { _ in }))
    }

    public func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        guard gate.session == session, gate.isPermitted() else { return preview(for: session.selection) }
        let generation = generations[session.selection] ?? ManagedImageGeneration(makeService: makeService)
        generations[session.selection] = generation
        guard let store = generation.store(for: session, isPermitted: gate.isPermitted) else {
            let state = ImageGenerationState(owner: UUID(), availability: .unsupported, retaining: generation.state)
            return AnyView(ImageGenerationContent(state: state, send: { _ in }))
        }
        return AnyView(ImageGenerationStoreContent(store: store))
    }

    public func stop(_ session: ModelDemoSession) async throws {
        try await generations[session.selection]?.release(session)
    }
}
