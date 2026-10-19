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
        var model: String?
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
                Text("The camera opens when you start searching.").font(.footnote).foregroundStyle(.secondary)
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
            .sheet(
                isPresented: Binding(
                    get: { store.state.showsCamera },
                    set: { if !$0, store.state.showsCamera { store.send(.stop) } })
            ) {
                ObjectFinderCameraSheet(
                    store: store, preview: preview, model: model ?? store.state.folder?.lastPathComponent)
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
        init(store: Store<ObjectFinderFeature>, preview: ObjectFinderPreview, model: String? = nil) {
            self.init(store: store, preview: preview, model: model, folderSettings: { EmptyView() })
        }
    }

    private struct ObjectFinderCameraSheet: View {
        let store: Store<ObjectFinderFeature>
        let preview: ObjectFinderPreview
        let model: String?

        var body: some View {
            NavigationStack {
                ScrollView {
                    VStack(spacing: 0) {
                        ObjectFinderLivePreview(
                            source: preview, minimumSimilarity: store.state.minimumSimilarity,
                            expectedGeneration: store.state.phase == .searching
                                ? store.state.frame?.captureGeneration : nil,
                            rotation: store.state.orientation.angle
                        )
                        .containerRelativeFrame(.vertical) { length, _ in length * 0.55 }
                        .background(.black)
                        .overlay {
                            if store.state.phase != .searching {
                                ProgressView("Preparing the search…").tint(.white).foregroundStyle(.white)
                            }
                        }
                        .accessibilityLabel("Live camera preview")
                        VStack(alignment: .leading, spacing: 14) {
                            queryEditor
                            ObjectFinderMetrics(state: store.state, model: model)
                        }
                        .padding()
                    }
                }
                .navigationTitle(Text(verbatim: store.state.submittedQuery))
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button(role: .close) { store.send(.stop) }
                    }
                }
            }
            .presentationDragIndicator(.visible)
        }

        private var queryEditor: some View {
            HStack {
                TextField(
                    "Type any object name",
                    text: Binding(get: { store.state.query }, set: { store.send(.queryChanged($0)) })
                )
                .textFieldStyle(.roundedBorder)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .submitLabel(.search)
                .onSubmit { store.send(.applyQuery) }
                .accessibilityLabel("Object to find")
                .disabled(store.state.phase == .stopping)
                if store.state.canApplyQuery {
                    Button("Search this text", systemImage: "arrow.clockwise") { store.send(.applyQuery) }
                        .buttonStyle(.bordered)
                }
            }
        }
    }

    private struct ObjectFinderMetrics: View {
        let state: ObjectFinderState
        let model: String?

        var body: some View {
            VStack(alignment: .leading, spacing: 14) {
                Grid(alignment: .topLeading, horizontalSpacing: 20, verticalSpacing: 12) {
                    GridRow {
                        metric(
                            "Analysis time", value: state.log.first.map { milliseconds($0.seconds) },
                            detail: state.averageSeconds.map { Text("Average \(milliseconds($0))") })
                        metric(
                            "Speed",
                            value: state.framesPerSecond.map { $0.formatted(.number.precision(.fractionLength(1))) },
                            detail: Text("analyses/s"))
                    }
                    GridRow {
                        metric(
                            "Memory", value: state.log.first?.memory.map { bytes($0.footprintBytes) },
                            detail: state.peakFootprintBytes.map { Text("Peak \(bytes($0))") })
                        metric(
                            "Available memory", value: state.log.first?.memory.map { bytes($0.availableBytes) },
                            detail: nil)
                    }
                }
                if let model {
                    LabeledContent("Model") {
                        Text(verbatim: model).font(.caption.monospaced()).lineLimit(1).truncationMode(.middle)
                    }
                    .font(.caption)
                }
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Similarity log").font(.subheadline.bold())
                        Spacer()
                        Text("Match threshold").font(.caption).foregroundStyle(.secondary)
                        Text(state.minimumSimilarity, format: .number.precision(.fractionLength(2)))
                            .font(.caption.monospacedDigit())
                    }
                    if state.log.isEmpty {
                        Text("Waiting for the first analysis…").font(.caption).foregroundStyle(.secondary)
                    }
                    ForEach(Array(state.log.enumerated()), id: \.offset) { offset, frame in
                        logRow(number: state.analyzedFrames - offset, frame: frame)
                    }
                    Text("Each value is the cosine similarity between the text vector and a region's image vector.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }

        private func metric(_ label: LocalizedStringKey, value: String?, detail: Text?) -> some View {
            VStack(alignment: .leading, spacing: 2) {
                Text(label).font(.caption).foregroundStyle(.secondary)
                Text(verbatim: value ?? "–").font(.title3.monospacedDigit())
                if let detail { detail.font(.caption2).foregroundStyle(.secondary) }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
        }

        private func logRow(number: Int, frame: ObjectFinderFrame) -> some View {
            HStack(spacing: 12) {
                Text(verbatim: "#\(number)").foregroundStyle(.secondary)
                Text(verbatim: milliseconds(frame.seconds))
                ForEach(Array(frame.similarities.sorted(by: >).enumerated()), id: \.offset) { _, value in
                    Text(value, format: .number.precision(.fractionLength(2)))
                        .fontWeight(value >= state.minimumSimilarity ? .bold : .regular)
                        .foregroundStyle(
                            value >= state.minimumSimilarity ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                }
            }
            .font(.caption.monospacedDigit())
            .accessibilityElement(children: .combine)
        }

        private func milliseconds(_ seconds: Double) -> String {
            (seconds * 1000).formatted(.number.precision(.fractionLength(0))) + " ms"
        }

        private func bytes(_ value: UInt64) -> String {
            Int64(clamping: value).formatted(.byteCount(style: .memory))
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
