#if os(iOS)
    import FeatureStore
    import Foundation
    import ObjectFinderCore
    import ObjectFinderFeature
    import ObjectFinderRuntime
    import SwiftUI
    import UIKit

    struct ObjectFinderContent<FolderSettings: View>: View {
        let store: Store<ObjectFinderFeature>
        let preview: ObjectFinderPreview
        var chooseFolder: (() -> Void)?
        @ViewBuilder var folderSettings: FolderSettings
        @FocusState private var editingQuery: Bool
        @Environment(\.scenePhase) private var scenePhase
        @Environment(\.openURL) private var openURL

        var body: some View {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Object to find").font(.headline)
                    TextField(
                        "Type any object name",
                        text: Binding(get: { store.state.query }, set: { store.send(.queryChanged($0)) })
                    )
                    .textFieldStyle(.plain)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .padding(12)
                    .frame(maxWidth: .infinity, minHeight: 48, alignment: .leading)
                    .background(.background, in: RoundedRectangle(cornerRadius: 12))
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(.secondary.opacity(0.5)))
                    .focused($editingQuery)
                    .submitLabel(.search)
                    .onSubmit(start)
                    .accessibilityIdentifier("object-finder-query")
                    .accessibilityLabel("Object to find")
                    .disabled(store.state.phase == .stopping)
                    Menu {
                        ForEach(ObjectFinderAssembly.examples, id: \.self) { text in
                            Button {
                                store.send(.queryChanged(text))
                                editingQuery = true
                            } label: {
                                Text(verbatim: text)
                            }
                        }
                    } label: {
                        Label("Fill an example", systemImage: "text.badge.plus")
                    }
                    .disabled(store.state.phase == .stopping)
                }
                controls
                cameraImage
                status
                if let failure = store.state.failure { failureView(failure) }
                DisclosureGroup("Search settings") {
                    VStack(alignment: .leading, spacing: 10) {
                        folderSettings
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
                if store.state.phase == .idle && !store.state.releasePending {
                    Button("Start searching", systemImage: "viewfinder", action: start)
                        .buttonStyle(.borderedProminent).disabled(!store.state.canStart)
                } else {
                    Button {
                        store.send(.stop)
                    } label: {
                        if store.state.restartAfterStop {
                            Text("Cancel change")
                        } else {
                            Label("Stop", systemImage: "stop.fill")
                        }
                    }
                    .buttonStyle(.borderedProminent).disabled(!store.state.canStop)
                }
                if store.state.canApplyQuery {
                    Button("Search this text", systemImage: "arrow.clockwise", action: start)
                        .buttonStyle(.bordered)
                }
                if store.state.folder == nil, let chooseFolder {
                    Button("Open model folder…", systemImage: "folder", action: chooseFolder)
                }
                Spacer()
            }
        }

        private var cameraImage: some View {
            ObjectFinderLivePreview(
                source: preview, minimumSimilarity: store.state.minimumSimilarity,
                expectedGeneration: store.state.phase == .searching ? store.state.frame?.captureGeneration : nil,
                rotation: store.state.orientation.angle
            )
            .frame(height: 360)
            .background(.black)
            .overlay {
                if store.state.phase == .idle || store.state.phase == .stopping {
                    ContentUnavailableView(
                        "Find things around you", systemImage: "viewfinder",
                        description: Text("The camera opens when you start searching.")
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.background)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .accessibilityLabel("Live camera preview")
        }

        @ViewBuilder
        private var status: some View {
            switch store.state.phase {
            case .starting: ProgressView("Preparing the search…")
            case .stopping: ProgressView("Stopping…")
            case .searching:
                Label("Searching…", systemImage: "viewfinder")
                Text(verbatim: store.state.submittedQuery).font(.headline)
                Text("The camera stays live while matches update.").font(.caption).foregroundStyle(
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
                case .notPermitted: Text("The selected model is not ready yet. Try again.")
                case .operation(let detail):
                    Text("Could not start or continue the search.")
                    Text(verbatim: detail).font(.caption).textSelection(.enabled)
                }
            }
            .foregroundStyle(.red)
        }

        private func start() {
            editingQuery = false
            store.send(store.state.phase == .idle ? .start : .applyQuery)
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

    extension ObjectFinderContent where FolderSettings == EmptyView {
        init(store: Store<ObjectFinderFeature>, preview: ObjectFinderPreview) {
            self.init(store: store, preview: preview, folderSettings: { EmptyView() })
        }
    }

    private struct ObjectFinderLivePreview: UIViewRepresentable {
        let source: ObjectFinderPreview
        let minimumSimilarity: Double
        let expectedGeneration: Int?
        let rotation: Double

        func makeUIView(context: Context) -> PreviewView { PreviewView(source: source) }

        func updateUIView(_ view: PreviewView, context: Context) {
            view.minimumSimilarity = minimumSimilarity
            view.expectedGeneration = expectedGeneration
            view.rotation = rotation
            view.applyPreviewState()
            view.setNeedsLayout()
        }

        static func dismantleUIView(_ view: PreviewView, coordinator: ()) {
            view.source.detach(from: view.layer)
        }

        final class PreviewView: UIView {
            let source: ObjectFinderPreview
            var minimumSimilarity = 0.67
            var expectedGeneration: Int?
            var rotation = 90.0

            init(source: ObjectFinderPreview) {
                self.source = source
                super.init(frame: .zero)
                source.attach(to: layer)
            }

            required init?(coder: NSCoder) { nil }

            override func layoutSubviews() {
                super.layoutSubviews()
                applyPreviewState()
            }

            func applyPreviewState() {
                source.layout(
                    in: bounds, minimumSimilarity: minimumSimilarity, expectedGeneration: expectedGeneration,
                    rotation: rotation)
            }
        }

    }
#endif
