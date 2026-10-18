import FeatureStore
import ModelLibraryFeature
import SwiftUI

#if os(iOS)
    import ModelLibraryUI
#endif

public struct ModelLibraryRendering {
    private let registry: ModelDemoAdapterRegistry
    private let runner: ModelLibraryRunner
    private let library: Store<ModelLibraryFeature>

    public init(registry: ModelDemoAdapterRegistry, runner: ModelLibraryRunner, library: Store<ModelLibraryFeature>) {
        self.registry = registry
        self.runner = runner
        self.library = library
    }

    public static func actions(
        entering phase: ScenePhase, capabilities: [DemoCapability: CapabilityState]
    ) -> [ModelLibraryAction] {
        switch phase {
        case .background: [.suspendDemo]
        case .active:
            capabilities.sorted { $0.key.rawValue < $1.key.rawValue }.map { .capabilityChanged($0.key, $0.value) }
        default: []
        }
    }

    public func content(for session: ModelDemoSession) -> AnyView {
        guard let adapter = registry.adapter(for: session.selection.demoID) else {
            return AnyView(Text("Demo execution is not connected in this build.", bundle: .module))
        }
        let content = adapter.content(for: session, gate: runner.gate(for: session, in: library))
        return AnyView(content.onAppear { [runner] in runner.provide(session) })
    }

    public func preview(for selection: DemoSelection) -> AnyView {
        registry.adapter(for: selection.demoID)?.preview(for: selection)
            ?? AnyView(DemoDescriptionPreview(demoID: selection.demoID))
    }
}

#if os(iOS)
    extension ModelLibraryRendering: ModelDemoRendering {}
#endif

private struct DemoDescriptionPreview: View {
    let demoID: String

    private var labels: (input: String, action: String) {
        switch demoID {
        case "comparison": ("Product description", "Compare")
        case "extraction": ("Text to extract", "Extract")
        case "inspection": ("Inspection answer", "Start inspection")
        case "image-search": ("Search query", "Search images")
        case "object-finder": ("What to find", "Start camera")
        default: ("Input", "Run demo")
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(LocalizedStringKey(labels.input), bundle: .module).font(.headline)
            TextField("", text: .constant(""), axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .accessibilityLabel(Text(LocalizedStringKey(labels.input), bundle: .module))
            Button {
            } label: {
                Text(LocalizedStringKey(labels.action), bundle: .module)
            }
            .buttonStyle(.borderedProminent)
        }
        .disabled(true)
    }
}
