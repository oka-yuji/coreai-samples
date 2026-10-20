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
        _ = runner.gate(for: session, in: library)
        let content: AnyView
        if library.state.demoPreparation == .available, let gate = runner.bind(session, in: library) {
            content = adapter.content(for: session, gate: gate)
        } else {
            content = AnyView(DemoPreparationView(session: session, library: library))
        }
        return AnyView(
            content.onAppear { [library] in
                if library.state.demoPreparation == .pending { library.send(.requestDemoPreparation(session.id)) }
            })
    }

    public func preview(for selection: DemoSelection) -> AnyView {
        registry.adapter(for: selection.demoID)?.preview(for: selection)
            ?? AnyView(DemoDescriptionPreview(demoID: selection.demoID))
    }
}

private struct DemoPreparationView: View {
    let session: ModelDemoSession
    let library: Store<ModelLibraryFeature>

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if [.occupied, .permissionRequired].contains(library.state.demoPreparation) {
                if library.state.demoPreparation == .occupied {
                    Text("Another window is using a model.", bundle: .module).font(.headline)
                    Text("Close its demo before trying again.", bundle: .module)
                } else {
                    Text("Allow the required permission, then try again.", bundle: .module)
                }
                Button {
                    library.send(.requestDemoPreparation(session.id))
                } label: {
                    Text("Try again", bundle: .module)
                }
                .buttonStyle(.bordered)
            } else if library.state.demoPreparation == .unavailable {
                Text("Demo execution is not connected in this build.", bundle: .module)
            } else {
                ProgressView { Text("Preparing the demo…", bundle: .module) }
            }
        }
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
        case "video-search": ("Search query", "Search video")
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
