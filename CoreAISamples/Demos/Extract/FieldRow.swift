import ExtractFeature
import GLiNERCore
import SwiftUI

/// One field of the form: its state, its candidates with their scores, the correction cues near them, and the
/// person's choice. Nothing is chosen by default. A label shows its spans and their scores only.
struct FieldRow: View {
    let field: FormField
    let isLabel: Bool
    let color: Color
    /// The field in the current result; nil before it arrives.
    let mapping: FieldMapping?
    let confirmation: ExtractChoice?
    let cues: [CorrectionCue]
    let selected: Bool
    let send: (ExtractAction) -> Void
    let edit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                RoundedRectangle(cornerRadius: 3).fill(color).frame(width: 12, height: 12)
                Text(field.name).font(.headline)
                Text(isLabel ? field.description ?? "" : kindDescription).font(.caption).foregroundStyle(.secondary)
                Spacer()
                status
            }
            ForEach(Array((mapping?.candidates ?? []).enumerated()), id: \.offset) { _, candidate in
                candidateLine(candidate)
            }
            if !cues.isEmpty {
                Label(
                    "Correction cue (rule): " + cues.map(\.word).joined(separator: ", "),
                    systemImage: "exclamationmark.triangle"
                )
                .font(.caption)
                .foregroundStyle(.orange)
            }
            HStack(spacing: 8) {
                if !isLabel {
                    Menu("Confirm") { choices }
                        .fixedSize()
                    Button("Clear") { send(.unconfirm(field: field.key)) }
                        .disabled(confirmation == nil)
                }
                Spacer()
                Button("Edit") { edit() }
                    .buttonStyle(.link)
            }
            .controlSize(.small)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(selected ? color.opacity(0.14) : Color(nsColor: .controlBackgroundColor))
        )
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(selected ? color : Color.secondary.opacity(0.25)))
        .contentShape(Rectangle())
        .onTapGesture { send(.select(field: selected ? nil : field.key)) }
    }

    private var kindDescription: String {
        switch field.kind {
        case .number(let unit): unit.isEmpty ? "Number" : "Number (\(unit))"
        case .list(let options, _): "List: " + options.map(\.name).joined(separator: " / ")
        }
    }

    @ViewBuilder private var status: some View {
        if let confirmation {
            Text("Confirmed: \(Self.describe(confirmation))").foregroundStyle(.green)
        } else if let mapping {
            if mapping.candidates.isEmpty {
                Text("Not detected").foregroundStyle(.secondary)
            } else if isLabel {
                Text(mapping.candidates.count == 1 ? "1 span" : "\(mapping.candidates.count) spans").foregroundStyle(
                    .blue)
            } else {
                Text(mapping.candidates.count == 1 ? "Candidate" : "Candidate (\(mapping.candidates.count))")
                    .foregroundStyle(.blue)
            }
        } else {
            Text("—").foregroundStyle(.secondary)
        }
    }

    private func candidateLine(_ candidate: FieldCandidate) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Text("\u{201C}\(candidate.span.text)\u{201D}").fontWeight(.medium)
                Text("span \(candidate.span.probability, format: .number.precision(.fractionLength(2)))")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                if let number = candidate.number, !isLabel {
                    Text("→ \(number.value.formatted()) \(number.normalizedUnit ?? "")").foregroundStyle(.secondary)
                }
            }
            if !candidate.suggestions.isEmpty {
                HStack(spacing: 6) {
                    Text(
                        candidate.suggestions.map {
                            "\($0.option) \($0.probability.formatted(.number.precision(.fractionLength(2))))"
                        }
                        .joined(separator: " · ")
                    )
                    .monospacedDigit()
                    Text("score (uncalibrated)").foregroundStyle(.secondary)
                }
                .font(.caption)
            }
        }
        .font(.callout)
    }

    @ViewBuilder private var choices: some View {
        switch field.kind {
        case .list(let options, _):
            ForEach(options, id: \.name) { option in
                Button(option.name) { send(.confirm(field: field.key, choice: .option(name: option.name))) }
            }
            Divider()
            Button("No matching option") { send(.confirm(field: field.key, choice: .noMatch)) }
        case .number:
            let values = (mapping?.candidates ?? []).map(\.span.text).reduce(into: [String]()) { values, text in
                if !values.contains(text) { values.append(text) }
            }
            ForEach(values, id: \.self) { value in
                Button(value) { send(.confirm(field: field.key, choice: .value(text: value))) }
            }
            Divider()
            Button("None of these") { send(.confirm(field: field.key, choice: .noMatch)) }
        }
        Button("On hold") { send(.confirm(field: field.key, choice: .hold)) }
    }

    static func describe(_ choice: ExtractChoice) -> String {
        switch choice {
        case .option(let name): name
        case .value(let text): text
        case .noMatch: "No match"
        case .hold: "On hold"
        }
    }
}
