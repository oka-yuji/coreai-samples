#if os(macOS)
    import Foundation
    import SpeechCore
    import SpeechFeature
    import SwiftUI
    import UniformTypeIdentifiers

    struct MacSpeechReferenceControls: View {
        static let activePhases: Set<SpeechState.Phase> = [
            .preparingRecording, .recording, .finishingRecording, .importingReference,
        ]
        static let microphoneSettings = URL(
            string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")

        let state: SpeechState
        let send: (SpeechAction) -> Void
        @State private var choice: MacSpeechAudioChoice?
        @State private var choices = 0
        @Environment(\.locale) private var locale

        var body: some View {
            VStack(alignment: .leading, spacing: 8) {
                Text("Reference voice (optional)").font(.headline)
                Text(
                    "Record your voice or choose an audio file. The text is read in that voice, and the voice description is not used."
                )
                .font(.callout).foregroundStyle(.secondary)
                if Self.activePhases.contains(state.phase) {
                    progress
                } else {
                    ViewThatFits(in: .horizontal) {
                        HStack { inputButtons }
                        VStack(alignment: .leading) { inputButtons }
                    }
                }
                Text(
                    "Use 5 to 10 seconds of your own voice or a voice you have permission to use. Recording stops automatically after 10 seconds."
                )
                .font(.caption).foregroundStyle(.secondary)
                if let message = state.referenceMessage {
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill").accessibilityHidden(true)
                        RuntimeMessageText(message)
                    }
                    .font(.callout).foregroundStyle(.orange)
                    if state.microphoneDenied, let settings = Self.microphoneSettings {
                        Link("Open microphone settings", destination: settings).font(.callout)
                    }
                }
                if let reference = state.reference { selected(reference) }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            .accessibilityElement(children: .contain)
            .accessibilityLabel(Text("Reference voice (optional)"))
            .background {
                if let choice {
                    MacSpeechAudioPicker(choice: choice, settle: settle, send: send).id(choice.id)
                }
            }
        }

        @ViewBuilder private var progress: some View {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { progressContent }
                VStack(alignment: .leading, spacing: 8) { progressContent }
            }
        }

        @ViewBuilder private var progressContent: some View {
            HStack(spacing: 6) {
                if state.phase == .recording {
                    Image(systemName: "mic.fill").foregroundStyle(.red).accessibilityHidden(true)
                } else {
                    ProgressView().controlSize(.small)
                }
                Text(verbatim: SpeechForm.phaseText(state.phase, locale: locale))
            }
            .accessibilityElement(children: .combine)
            if state.phase == .recording {
                Button("Finish recording", systemImage: "stop.circle.fill") { send(.finishReferenceRecording) }
                    .buttonStyle(.borderedProminent).tint(.red)
                    .accessibilityIdentifier("speech.reference.finish")
            }
            Button(cancelTitle, role: .cancel) { send(.stop) }
                .keyboardShortcut(.cancelAction)
                .accessibilityIdentifier("speech.reference.cancel")
        }

        @ViewBuilder private var inputButtons: some View {
            Button("Record voice", systemImage: "mic.fill") { send(.recordReference) }
                .buttonStyle(.bordered)
                .disabled(!state.canSelectReference)
                .accessibilityIdentifier("speech.reference.record")
            Button("Choose audio file", systemImage: "waveform.badge.plus") {
                choices += 1
                choice = MacSpeechAudioChoice(id: choices, session: state.session)
            }
            .buttonStyle(.bordered)
            .disabled(!state.canSelectReference)
            .accessibilityIdentifier("speech.reference.import")
        }

        private func selected(_ reference: SpeechReferenceAudio) -> some View {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { selectedContent(reference) }
                VStack(alignment: .leading, spacing: 6) { selectedContent(reference) }
            }
        }

        @ViewBuilder private func selectedContent(_ reference: SpeechReferenceAudio) -> some View {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: "waveform.circle.fill").foregroundStyle(.tint).accessibilityHidden(true)
                switch reference.source {
                case .recording: Text("Recorded voice")
                case .file(let name): Text(verbatim: name).lineLimit(1).truncationMode(.middle)
                }
                Text("\(reference.seconds, format: .number.precision(.fractionLength(1))) s reference")
                    .foregroundStyle(.secondary)
            }
            .font(.callout)
            .accessibilityElement(children: .combine)
            Button("Listen to reference", systemImage: "play.circle") { send(.playReference) }
                .disabled(!state.canPreviewReference)
                .accessibilityIdentifier("speech.reference.preview")
            Button("Remove reference", systemImage: "xmark.circle") { send(.clearReference) }
                .disabled(!state.editable)
                .accessibilityIdentifier("speech.reference.remove")
        }

        private var cancelTitle: LocalizedStringKey {
            state.phase == .importingReference ? "Cancel" : "Cancel recording"
        }

        private func settle(_ id: Int) -> Bool {
            guard choice?.id == id else { return false }
            choice = nil
            return true
        }
    }

    struct MacSpeechAudioChoice: Identifiable, Equatable {
        let id: Int
        let session: Int
    }

    private struct MacSpeechAudioPicker: View {
        let choice: MacSpeechAudioChoice
        let settle: (Int) -> Bool
        let send: (SpeechAction) -> Void
        @State private var choosing = true

        var body: some View {
            Color.clear.frame(width: 0, height: 0)
                .allowsHitTesting(false)
                .accessibilityHidden(true)
                .fileImporter(isPresented: $choosing, allowedContentTypes: [.audio], allowsMultipleSelection: false) {
                    result in
                    guard settle(choice.id) else { return }
                    switch result {
                    case .success(let urls):
                        if let url = urls.first { send(.importReference(choice.session, url)) }
                    case .failure(let error):
                        if (error as? CocoaError)?.code != .userCancelled {
                            send(.referenceSelectionFailed(choice.session, error.localizedDescription))
                        }
                    }
                } onCancellation: {
                    _ = settle(choice.id)
                }
        }
    }
#endif
