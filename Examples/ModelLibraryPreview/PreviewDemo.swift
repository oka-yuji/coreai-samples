import ModelLibraryFeature
import ModelLibraryUI
import SwiftUI

@MainActor
final class PreviewRenderer: ModelDemoRendering {
    private let inputs: [DemoSelection: PreviewDemoState]

    init(catalog: ModelCatalog) {
        inputs = Dictionary(
            uniqueKeysWithValues: catalog.models.flatMap { model in
                model.demos.map { link in
                    (DemoSelection(modelID: model.id, demoID: link.demoID), PreviewDemoState())
                }
            })
    }

    func content(for session: ModelDemoSession) -> AnyView { preview(for: session.selection) }

    func preview(for selection: DemoSelection) -> AnyView {
        if let state = inputs[selection] { AnyView(PreviewDemo(state: state)) } else { AnyView(EmptyView()) }
    }
}

@Observable
@MainActor
final class PreviewDemoState {
    var text = "A short sample input."
    var result = ""
}

struct PreviewDemo: View {
    @Bindable var state: PreviewDemoState

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            TextField("Sample input", text: $state.text, axis: .vertical).textFieldStyle(.roundedBorder)
            Button("Show sample result") { state.result = "Preview only. No inference was performed." }
                .buttonStyle(.borderedProminent)
            if !state.result.isEmpty { Text(state.result) }
        }
    }
}
