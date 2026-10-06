import ExtractFeature
import FeatureStore
import Observation
import SwiftUI

enum Demo: String, CaseIterable, Identifiable {
    case chat
    case agent
    case extract
    case models

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chat: "Chat"
        case .agent: "Agent"
        case .extract: "Extract"
        case .models: "Models"
        }
    }

    var summary: String {
        switch self {
        case .chat: "Talk to a local model through Foundation Models"
        case .agent: "Search the web and run multi-step tasks with tool calls"
        case .extract: "Technical verification: type labels and see the spans GLiNER finds in a text"
        case .models: "Get, open and prepare model bundles"
        }
    }

    var systemImage: String {
        switch self {
        case .chat: "bubble.left.and.bubble.right"
        case .agent: "sparkles"
        case .extract: "text.magnifyingglass"
        case .models: "square.and.arrow.down"
        }
    }
}

@MainActor
@Observable
final class DemoNavigator {
    var selection: Demo? = .chat
}

struct DemoRootView: View {
    @State private var navigator = DemoNavigator()
    @State private var chat = ChatViewModel()
    @State private var agent = AgentViewModel()
    @State private var models = ModelsViewModel()
    @State private var extract = Store<ExtractFeature>(
        initialState: ExtractState(
            form: ExtractResources.form, labels: ExtractResources.examples.labels,
            lastAssetPath: UserDefaults.standard.string(forKey: ExtractRunner.assetFolderKey),
            appVersion: ExtractRunner.appVersion),
        run: ExtractRunner(service: GLiNERExtractService(correctionWords: ExtractResources.correctionWords)).run)

    var body: some View {
        @Bindable var navigator = navigator
        NavigationSplitView {
            List(selection: $navigator.selection) {
                ForEach(Demo.allCases) { demo in
                    NavigationLink(value: demo) {
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(demo.title)
                                Text(demo.summary)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        } icon: {
                            Image(systemName: demo.systemImage)
                        }
                    }
                }
            }
            .navigationTitle("Demos")
            .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        } detail: {
            switch navigator.selection {
            case .chat?: ChatView()
            case .agent?: AgentView()
            case .extract?: ExtractView(store: extract)
            case .models?: ModelsView()
            case nil: ContentUnavailableView("Select a demo", systemImage: "square.grid.2x2")
            }
        }
        .environment(navigator)
        .environment(chat)
        .environment(agent)
        .environment(models)
        .task {
            GUIDriver.shared.attach(navigator: navigator, chat: chat, agent: agent, models: models, extract: extract)
        }
    }
}
