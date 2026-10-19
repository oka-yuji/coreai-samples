#if os(iOS)
    import ExtractFeature
    import GLiNERCore
    import SwiftUI

    struct PhoneExtractContent: View {
        let state: ExtractState
        @Binding var text: String
        let send: @MainActor (ExtractAction) -> Void

        var body: some View {
            VStack(alignment: .leading, spacing: 12) {
                Text("Note").font(.headline)
                TextField("Paste or type a note.", text: $text, axis: .vertical)
                    .lineLimit(3...8)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel(Text("Note"))
                Button("Extract") {
                    send(.textChanged(text))
                    send(.extractRequested)
                }
                .buttonStyle(.borderedProminent)
                .disabled(text.allSatisfy(\.isWhitespace))
                status
                if let mapping = state.result?.mapping {
                    Text("Labels").font(.headline)
                    ForEach(mapping.fields, id: \.field.key, content: labelRow)
                    if let identity = state.resourceIdentity {
                        LabeledContent {
                            Text(verbatim: identity).textSelection(.enabled)
                        } label: {
                            Text("Model")
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }

        @ViewBuilder private var status: some View {
            switch state.phase {
            case .extracting:
                HStack(spacing: 8) {
                    ProgressView()
                    if case .opening = state.asset { Text("Loading…") } else { Text("Extracting…") }
                }
                .foregroundStyle(.secondary)
            case .failed(let message):
                Text(message).foregroundStyle(.red)
            case .ready, .idle:
                if let timing = state.result?.timing {
                    Text(ExtractRunner.describe(timing)).font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
            }
        }

        private func labelRow(_ label: FieldMapping) -> some View {
            VStack(alignment: .leading, spacing: 4) {
                Text(label.field.name).font(.subheadline.bold())
                if label.candidates.isEmpty {
                    Text("Not detected").foregroundStyle(.secondary)
                }
                ForEach(Array(label.candidates.enumerated()), id: \.offset) { _, candidate in
                    Text(
                        "\u{201C}\(candidate.span.text)\u{201D} span "
                            + candidate.span.probability.formatted(.number.precision(.fractionLength(2)))
                    )
                }
            }
        }
    }
#endif
