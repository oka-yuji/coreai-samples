import FeatureStore
import Foundation
import ImageGenerationCore
import ImageGenerationFeature
import SwiftUI
import UniformTypeIdentifiers

public struct ImageGenerationContent: View {
    private enum Field: Hashable { case prompt, seed }
    public let state: ImageGenerationState
    public let send: (ImageGenerationAction) -> Void
    @State private var choosingReference = false
    @Environment(\.locale) private var locale
    @FocusState private var focusedField: Field?

    public init(state: ImageGenerationState, send: @escaping (ImageGenerationAction) -> Void) {
        self.state = state
        self.send = send
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text(state.modelName).font(.headline)
                Text("Create a 512-square image, or edit one reference image.", bundle: .module)
                    .foregroundStyle(.secondary)
                if state.availability == .unsupported {
                    Label {
                        Text("Generation is not available for this model on this device yet.", bundle: .module)
                    } icon: {
                        Image(systemName: "info.circle")
                    }
                } else if state.availability == .needsModel {
                    Text("Model not ready", bundle: .module).foregroundStyle(.secondary)
                }
            }
            promptSection
            referenceSection
            operationSection
            if let result = state.result { resultSection(result) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .fileImporter(isPresented: $choosingReference, allowedContentTypes: [.image], allowsMultipleSelection: false) {
            result in
            switch result {
            case .success(let urls): if let first = urls.first { send(.chooseReference(first)) }
            case .failure(let error):
                if (error as? CocoaError)?.code != .userCancelled { send(.referenceSelectionFailed) }
            }
        }
        .sheet(
            item: Binding(
                get: { state.proposedReference },
                set: { value in
                    if value == nil, let preview = state.proposedReference { send(.dismissReference(preview.id)) }
                }
            )
        ) { preview in
            ImageReferenceConfirmation(preview: preview, send: send)
                .environment(\.locale, locale)
        }
        .background {
            if let ticket = state.export, let result = state.result, result.id == ticket.resultID {
                ImagePNGExportPresenter(ticket: ticket, result: result, send: send).id(ticket.id)
            }
        }
        .onAppear { send(.appeared) }
        .onDisappear { send(.ended) }
        .onChange(of: state.phase) { _, phase in if phase != .idle { focusedField = nil } }
    }

    private var promptSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                TextEditor(text: Binding(get: { state.prompt }, set: { send(.setPrompt($0)) }))
                    .frame(minHeight: 110)
                    .focused($focusedField, equals: .prompt)
                    .accessibilityLabel(Text("Prompt", bundle: .module))
                    .disabled(!state.editable)
                LabeledContent {
                    TextField(
                        "", text: Binding(get: { state.seed }, set: { send(.setSeed($0)) }),
                        prompt: Text("Automatic", bundle: .module)
                    )
                    .monospacedDigit()
                    .focused($focusedField, equals: .seed)
                    .accessibilityLabel(Text("Seed", bundle: .module))
                    .disabled(!state.editable)
                } label: {
                    Text("Seed", bundle: .module)
                }
                if !state.seedIsValid {
                    Text("Enter a whole number from 0 to 9223372036854775807.", bundle: .module)
                        .foregroundStyle(.red).font(.caption)
                }
                Text("An automatic seed is shown after starting and reused when you retry.", bundle: .module)
                    .font(.caption).foregroundStyle(.secondary)
                if state.prompt.utf8.count > 262_144 {
                    Text("The prompt is too long. Shorten it and try again.", bundle: .module)
                        .foregroundStyle(.red).font(.caption)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Text("Prompt", bundle: .module)
        }
    }

    private var referenceSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                if let reference = state.reference {
                    ImagePNGView(data: reference.cropped)
                        .frame(maxHeight: 180)
                        .accessibilityLabel(Text("Confirmed reference image", bundle: .module))
                    Button(role: .destructive) {
                        send(.removeReference)
                    } label: {
                        Label {
                            Text("Remove reference", bundle: .module)
                        } icon: {
                            Image(systemName: "xmark.circle")
                        }
                    }.disabled(!state.editable)
                }
                Button {
                    focusedField = nil
                    choosingReference = true
                } label: {
                    Label {
                        Text("Choose image", bundle: .module)
                    } icon: {
                        Image(systemName: "photo")
                    }
                }.disabled(!state.editable)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Text("Reference image", bundle: .module)
        }
    }

    private var operationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            if [.readingReference, .generating, .stopping].contains(state.phase) {
                if let step = generationStep {
                    ProgressView(value: Double(step), total: 8) {
                        Text("Generating image", bundle: .module)
                    } currentValueLabel: {
                        Text("Step \(min(step + 1, 8)) of 8", bundle: .module)
                    }
                    Text("Step progress covers image generation. Finishing the image follows.", bundle: .module)
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text(phaseLabel, bundle: .module)
                    }
                }
            }
            if let failure = state.failure {
                Label {
                    Text(ImageGenerationCopy.failure(failure), bundle: .module)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .foregroundStyle(.red)
            }
            if state.phase == .idle {
                Button {
                    send(.generate(id: UUID(), randomSeed: UInt64.random(in: 0...UInt64(Int64.max))))
                } label: {
                    Label {
                        Text(state.reference == nil ? "Generate image" : "Edit image", bundle: .module)
                    } icon: {
                        Image(systemName: "sparkles")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!state.canGenerate)
                Text("First-time preparation can take longer.", bundle: .module)
                    .font(.caption).foregroundStyle(.secondary)
            } else if state.canStop {
                Button {
                    send(.stop)
                } label: {
                    Text(state.phase == .cleanupFailed ? "Retry stop" : "Stop", bundle: .module)
                }.buttonStyle(.bordered)
            }
        }
    }

    private func resultSection(_ result: ImageGeneratedAsset) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                ImagePNGView(data: result.png)
                    .accessibilityLabel(Text("Generated image", bundle: .module))
                HStack {
                    Button {
                        focusedField = nil
                        send(.prepareExport(UUID()))
                    } label: {
                        Label {
                            Text("Save PNG", bundle: .module)
                        } icon: {
                            Image(systemName: "square.and.arrow.down")
                        }
                    }.disabled(!state.canExport)
                    ShareLink(
                        item: ImagePNGTransfer(data: result.png, filename: result.filename),
                        preview: SharePreview(
                            Text(
                                verbatim: Bundle.module.localizedString(
                                    forKey: "Generated image", value: nil, table: nil,
                                    localizations: [Locale.Language(identifier: locale.identifier)])),
                            image: ImagePNGView.image(result.png))
                    ) {
                        Label {
                            Text("Share", bundle: .module)
                        } icon: {
                            Image(systemName: "square.and.arrow.up")
                        }
                    }.disabled(!state.canExport)
                }
                if state.export != nil && state.exportDismissed {
                    Text("Saving PNG…", bundle: .module)
                } else if state.exportSucceeded {
                    Text("PNG saved.", bundle: .module).foregroundStyle(.secondary)
                }
                DisclosureGroup {
                    LabeledContent {
                        Text(String(result.seed)).textSelection(.enabled)
                    } label: {
                        Text("Seed", bundle: .module)
                    }
                    LabeledContent {
                        Text(state.modelName)
                    } label: {
                        Text("Model", bundle: .module)
                    }
                    LabeledContent {
                        Text(result.report.profileID).textSelection(.enabled)
                    } label: {
                        Text("Profile", bundle: .module)
                    }
                    LabeledContent {
                        Text(result.report.manifestSHA256).textSelection(.enabled)
                    } label: {
                        Text("Model file version", bundle: .module)
                    }
                    LabeledContent {
                        Text(ImageGenerationProfile.revision).textSelection(.enabled)
                    } label: {
                        Text("Source revision", bundle: .module)
                    }
                    LabeledContent {
                        Text(result.report.totalSeconds, format: .number.precision(.fractionLength(1)))
                    } label: {
                        Text("Elapsed seconds", bundle: .module)
                    }
                    if let license = URL(
                        string:
                            "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/\(ImageGenerationProfile.revision)/LICENSE"
                    ) {
                        Link(destination: license) { Text("Qwen Research License", bundle: .module) }
                    }
                } label: {
                    Text("Image details", bundle: .module)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Text("Last completed image", bundle: .module)
        }
    }

    private var generationStep: Int? {
        guard state.phase == .generating else { return nil }
        if case .running(.denoiser, let step) = state.progress { return step }
        if state.progress == .releasing(.denoiser) { return 8 }
        return nil
    }

    private var phaseLabel: LocalizedStringKey {
        switch state.phase {
        case .readingReference: return "Preparing reference image"
        case .confirmingReference: return "Confirm the reference image"
        case .stopping: return "Stopping…"
        case .cleanupFailed: return "Stopping needs to be retried"
        case .idle: return "Ready"
        case .generating: break
        }
        switch state.progress {
        case .running(.vision, _), .running(.vaeEncoder, _): return "Reading reference image"
        case .running(.text, _): return "Reading the prompt"
        case .running(.vaeDecoder, _): return "Finishing the image"
        default: return "Preparing the model"
        }
    }
}

public struct ImageGenerationStoreContent: View {
    public let store: Store<ImageGenerationFeature>

    public init(store: Store<ImageGenerationFeature>) { self.store = store }

    public var body: some View { ImageGenerationContent(state: store.state, send: store.send) }
}
