#if os(iOS)
    import FeatureStore
    import Foundation
    import ObjectFinderCore
    import ObjectFinderFeature
    import ObjectFinderRuntime
    import SwiftUI
    import UIKit
    import UniformTypeIdentifiers

    @MainActor
    enum ObjectFinderAssembly {
        static func makeStore() -> Store<ObjectFinderFeature> {
            let prepared = URL.documentsDirectory.appending(path: "embeddinggemma2")
            let folder =
                LaunchArguments.value("--embedding-bundle").map { URL(filePath: $0) }
                ?? (FileManager.default.fileExists(atPath: prepared.appending(path: "embedding-manifest.json").path)
                    ? prepared : nil)
            return Store(
                initialState: ObjectFinderState(folder: folder),
                run: ObjectFinderRunner(service: LiveObjectFinderService()).run)
        }

        static var examples: [String] {
            guard let url = Bundle.main.url(forResource: "object-finder-examples", withExtension: "json"),
                let data = try? Data(contentsOf: url), let values = try? JSONDecoder().decode([String].self, from: data)
            else { return [] }
            return values
        }
    }

    struct ObjectFinderView: View {
        let store: Store<ObjectFinderFeature>
        @State private var choosesModel = false
        @FocusState private var editingQuery: Bool
        @Environment(\.scenePhase) private var scenePhase
        @Environment(\.openURL) private var openURL

        var body: some View {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("Name an object, then point your camera around the room.")
                        .foregroundStyle(.secondary)
                    HStack {
                        TextField(
                            "What are you looking for?",
                            text: Binding(get: { store.state.query }, set: { store.send(.queryChanged($0)) })
                        )
                        .textFieldStyle(.roundedBorder)
                        .focused($editingQuery)
                        .submitLabel(.search)
                        .onSubmit(start)
                        Menu {
                            ForEach(ObjectFinderAssembly.examples, id: \.self) { text in
                                Button {
                                    store.send(.queryChanged(text))
                                } label: {
                                    Text(verbatim: text)
                                }
                            }
                        } label: {
                            Label("Examples", systemImage: "text.badge.plus")
                        }
                        .labelStyle(.iconOnly)
                    }
                    .disabled(store.state.phase != .idle)
                    controls
                    cameraImage
                    status
                    if let failure = store.state.failure { failureView(failure) }
                    DisclosureGroup("Search settings") {
                        VStack(alignment: .leading, spacing: 10) {
                            Button {
                                choosesModel = true
                            } label: {
                                Label(
                                    store.state.folder == nil ? "Open model folder…" : "Change model folder…",
                                    systemImage: "folder")
                            }
                            if let folder = store.state.folder {
                                Text(verbatim: folder.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                            }
                            HStack {
                                Text("Match threshold")
                                Spacer()
                                Text(store.state.minimumSimilarity, format: .number.precision(.fractionLength(2)))
                            }
                            Slider(
                                value: Binding(
                                    get: { store.state.minimumSimilarity }, set: { store.send(.thresholdChanged($0)) }),
                                in: 0.3...0.9, step: 0.01
                            )
                            .accessibilityLabel("Match threshold")
                            Text("Higher values show fewer candidates. Similarity is not a probability.").font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.top, 8)
                        .disabled(store.state.phase != .idle)
                    }
                    Text(
                        "Images are analyzed on this iPhone and are not saved. Small or obscured objects may be missed."
                    )
                    .font(.footnote).foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("Object finder")
            .fileImporter(isPresented: $choosesModel, allowedContentTypes: [.folder]) { result in
                switch result {
                case .success(let url): store.send(.folderSelected(url))
                case .failure(let error): store.send(.inputFailed(error.localizedDescription))
                }
            }
            .onAppear {
                store.send(.appeared)
                updateOrientation()
            }
            .onReceive(NotificationCenter.default.publisher(for: UIDevice.orientationDidChangeNotification)) { _ in
                updateOrientation()
            }
            .onDisappear { store.send(.disappeared) }
            .onChange(of: scenePhase) { _, value in
                if value == .background { store.send(.backgrounded) }
            }
        }

        private var controls: some View {
            HStack {
                if store.state.phase == .idle {
                    Button("Start searching", systemImage: "viewfinder", action: start)
                        .buttonStyle(.borderedProminent).disabled(!store.state.canStart)
                } else {
                    Button("Stop", systemImage: "stop.fill") { store.send(.stop) }
                        .buttonStyle(.borderedProminent).disabled(store.state.phase == .stopping)
                }
                if store.state.folder == nil {
                    Button("Open model folder…", systemImage: "folder") { choosesModel = true }
                }
                Spacer()
            }
        }

        @ViewBuilder
        private var cameraImage: some View {
            if let data = store.state.frame?.image, let image = UIImage(data: data) {
                Image(uiImage: image).resizable().aspectRatio(contentMode: .fit)
                    .overlay {
                        GeometryReader { geometry in
                            ForEach(store.state.matches) { candidate in
                                let box = candidate.region
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(.yellow, lineWidth: 3)
                                    .frame(
                                        width: geometry.size.width * box.width,
                                        height: geometry.size.height * box.height
                                    )
                                    .position(
                                        x: geometry.size.width * (box.x + box.width / 2),
                                        y: geometry.size.height * (box.y + box.height / 2))
                            }
                        }
                        .accessibilityHidden(true)
                    }
                    .accessibilityLabel("Analyzed camera image")
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            } else {
                ContentUnavailableView(
                    "Find things around you", systemImage: "viewfinder",
                    description: Text("The camera opens when you start searching.")
                )
                .frame(maxWidth: .infinity, minHeight: 240)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 16))
            }
        }

        @ViewBuilder
        private var status: some View {
            switch store.state.phase {
            case .starting: ProgressView("Preparing the search…")
            case .stopping: ProgressView("Stopping…")
            case .searching:
                Label(
                    store.state.matches.isEmpty ? "Searching…" : "Possible match",
                    systemImage: store.state.matches.isEmpty ? "viewfinder" : "checkmark.circle")
                Text(verbatim: store.state.submittedQuery).font(.headline)
                Text("The image and boxes update together after each analysis.").font(.caption).foregroundStyle(
                    .secondary)
            case .idle: EmptyView()
            }
        }

        @ViewBuilder
        private func failureView(_ failure: ObjectFinderFailure) -> some View {
            VStack(alignment: .leading, spacing: 8) {
                switch failure {
                case .cameraDenied:
                    Text("Allow camera access in Settings to find objects.")
                    Button("Open Settings") {
                        if let url = URL(string: UIApplication.openSettingsURLString) { openURL(url) }
                    }
                case .cameraUnavailable: Text("The rear camera is unavailable.")
                case .cameraInterrupted: Text("The camera was interrupted. Start again when it is available.")
                case .operation(let detail):
                    Text("Could not start or continue the search.")
                    Text(verbatim: detail).font(.caption).textSelection(.enabled)
                }
            }
            .foregroundStyle(.red)
        }

        private func start() {
            editingQuery = false
            store.send(.start)
        }

        private func updateOrientation() {
            let pose: ObjectCameraPose
            switch UIDevice.current.orientation {
            case .portrait: pose = .portrait
            case .portraitUpsideDown: pose = .portraitUpsideDown
            case .landscapeLeft: pose = .landscapeLeft
            case .landscapeRight: pose = .landscapeRight
            default: pose = .flat
            }
            store.send(.poseChanged(pose))
        }
    }
#endif
