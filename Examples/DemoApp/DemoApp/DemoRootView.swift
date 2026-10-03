import Observation
import SwiftUI

enum Demo: String, CaseIterable, Identifiable {
    case chat
    case models

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chat: "Chat"
        case .models: "Models"
        }
    }

    var summary: String {
        switch self {
        case .chat: "Talk to a local model through Foundation Models"
        case .models: "Get, open and prepare model bundles"
        }
    }

    var systemImage: String {
        switch self {
        case .chat: "bubble.left.and.bubble.right"
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
    @State private var models = ModelsViewModel()

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
            case .models?: ModelsView()
            case nil: ContentUnavailableView("Select a demo", systemImage: "square.grid.2x2")
            }
        }
        .environment(navigator)
        .environment(chat)
        .environment(models)
        .task { GUIDriver.shared.attach(navigator: navigator, chat: chat, models: models) }
    }
}
