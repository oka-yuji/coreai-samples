import RecipeFeature
import SwiftUI

struct RecipeModelView: View {
    let state: RecipeState
    let retry: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            Image(systemName: state.modelName == nil ? "cpu" : "checkmark.circle.fill")
                .font(.title2).foregroundStyle(state.modelName == nil ? Color.secondary : .green)
            VStack(alignment: .leading, spacing: 4) {
                Text("Recipe decision model").font(.caption).foregroundStyle(.secondary)
                Text(verbatim: "clef-flash-9b-decide-v1").font(.headline.monospaced())
                status.font(.caption)
            }
            Spacer(minLength: 8)
            if state.modelName == nil && !state.modelPath.isEmpty {
                Button("Retry", action: retry).disabled(state.busy)
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 12)
        .background(.background.secondary)
    }

    @ViewBuilder
    private var status: some View {
        switch state.phase {
        case .loading: Text("Loading the decision model...")
        case .stopping, .closing: Text("Waiting for the current operation to stop...")
        default:
            if let name = state.modelName {
                Text("Loaded: \(name)").foregroundStyle(.green)
            } else {
                Text("Download Clef-Flash from its heading in the sidebar.").foregroundStyle(.secondary)
            }
        }
    }
}
