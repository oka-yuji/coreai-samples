#if os(iOS)
    import Foundation
    import SpeechFeature
    import SwiftUI
    import UIKit
    import UniformTypeIdentifiers

    struct SpeechReferenceControls: View {
        let state: SpeechState
        let send: (SpeechAction) -> Void
        @State private var choosingAudio = false
        @State private var selectionSession: Int?

        var body: some View {
            VStack(alignment: .leading, spacing: 12) {
                Text("2. Choose a reference voice").font(.headline)
                Text(
                    "Record your voice or choose an audio file. The generated speech will use it as a voice reference."
                )
                .font(.callout).foregroundStyle(.secondary)
                if state.phase == .recording {
                    Label("Recording your reference voice…", systemImage: "mic.fill")
                        .foregroundStyle(.red)
                    Button("Finish recording", systemImage: "stop.circle.fill") { send(.finishReferenceRecording) }
                        .buttonStyle(.borderedProminent).tint(.red)
                        .accessibilityIdentifier("speech.reference.finish")
                    Button("Cancel recording", role: .cancel) { send(.stop) }
                } else if [.preparingRecording, .finishingRecording, .importingReference].contains(state.phase) {
                    ProgressView {
                        switch state.phase {
                        case .preparingRecording: Text("Preparing to record…")
                        case .finishingRecording: Text("Preparing your recorded voice…")
                        default: Text("Reading the reference audio…")
                        }
                    }
                    Button("Cancel", role: .cancel) { send(.stop) }
                } else {
                    ViewThatFits(in: .horizontal) {
                        HStack { inputButtons }
                        VStack(alignment: .leading) { inputButtons }
                    }
                }
                Text(
                    "Use a clear voice recording from 0.5 to 30 seconds. Recording stops automatically after 30 seconds."
                )
                .font(.caption).foregroundStyle(.secondary)
                if let message = state.referenceMessage {
                    Text(LocalizedStringKey(message)).font(.callout).foregroundStyle(.orange)
                }
                if let reference = state.reference {
                    HStack(alignment: .top) {
                        Image(systemName: "waveform.circle.fill").foregroundStyle(.tint)
                        VStack(alignment: .leading, spacing: 3) {
                            switch reference.source {
                            case .recording: Text("Recorded voice")
                            case .file(let name): Text(verbatim: name).lineLimit(2)
                            }
                            Text("\(reference.seconds, format: .number.precision(.fractionLength(1))) s reference")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    HStack {
                        Button("Listen to reference", systemImage: "play.fill") { send(.playReference) }
                            .disabled(!state.canPreviewReference)
                        Spacer()
                        Button("Remove reference", systemImage: "xmark.circle") { send(.clearReference) }
                            .disabled(!state.editable)
                    }
                } else {
                    Text("You can also leave this empty to use a generated voice.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let model = state.model, !model.supportsReferenceAudio {
                    Text("Choose the Reference voice model to enable recording and audio files.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                if state.microphoneDenied, let settings = URL(string: UIApplication.openSettingsURLString) {
                    Link("Open microphone settings", destination: settings)
                }
            }
            .fileImporter(isPresented: $choosingAudio, allowedContentTypes: [.audio]) { result in
                guard let session = selectionSession else { return }
                selectionSession = nil
                switch result {
                case .success(let url): send(.importReference(session, url))
                case .failure(let error):
                    if (error as? CocoaError)?.code != .userCancelled, (error as? URLError)?.code != .cancelled {
                        send(.referenceSelectionFailed(session, error.localizedDescription))
                    }
                }
            }
        }

        @ViewBuilder private var inputButtons: some View {
            Button("Record voice", systemImage: "mic.fill") { send(.recordReference) }
                .buttonStyle(.bordered)
                .disabled(!state.canSelectReference)
                .accessibilityIdentifier("speech.reference.record")
            Button("Choose audio file", systemImage: "waveform.badge.plus") {
                selectionSession = state.session
                choosingAudio = true
            }
            .buttonStyle(.bordered)
            .disabled(!state.canSelectReference)
            .accessibilityIdentifier("speech.reference.import")
        }
    }
#endif
