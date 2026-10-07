import DecideFeature
import SwiftUI

struct DecideResultRow: View {
    let row: DecideRow
    let previous: DecideRow?
    let changed: Bool
    let confirmation: String?
    let enabled: Bool
    let confirm: (String?) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text(row.key).font(.headline)
                Spacer()
                if changed { Text("Changed").font(.caption.bold()).foregroundStyle(.blue) }
                if row.reviewFirst {
                    Label("Review first", systemImage: "magnifyingglass")
                        .font(.caption.bold()).foregroundStyle(.orange)
                }
                Text(confirmation == nil ? "Unverified" : "Confirmed")
                    .font(.caption).foregroundStyle(confirmation == nil ? Color.secondary : Color.green)
            }
            LabeledContent("Model candidate") {
                Text(candidateLabel).fontWeight(.medium).textSelection(.enabled)
            }
            .font(.callout)
            if let score = row.expectedScore {
                LabeledContent("Expected score", value: score.formatted(.number.precision(.fractionLength(2))))
                    .font(.caption)
            }
            ForEach(row.probabilities) { probability in
                HStack(spacing: 10) {
                    Text(probability.label).font(.caption)
                        .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                    ProgressView(value: Double(probability.probability)).frame(width: 70)
                    Text(probability.probability.formatted(.percent.precision(.fractionLength(1))))
                        .font(.caption.monospacedDigit()).frame(width: 48, alignment: .trailing)
                }
            }
            Picker(
                "Confirmed by reviewer",
                selection: Binding<Int?>(
                    get: {
                        guard let confirmation else { return nil }
                        return row.probabilities.first { $0.option.utf8.elementsEqual(confirmation.utf8) }?.id
                    },
                    set: { id in confirm(row.probabilities.first { $0.id == id }?.option) }
                )
            ) {
                Text("Unverified").tag(Int?.none)
                ForEach(row.probabilities) { probability in
                    Text(probability.label).tag(Optional(probability.id))
                }
            }
            .font(.caption).disabled(!enabled)
            if let previous {
                DisclosureGroup("Compare previous result") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Previous candidate: \(previous.candidate)").font(.caption)
                        ForEach(previous.probabilities) { probability in
                            HStack {
                                Text(probability.label).lineLimit(2)
                                Spacer()
                                Text(probability.probability.formatted(.percent.precision(.fractionLength(1))))
                                    .monospacedDigit()
                            }
                            .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .padding(.top, 4)
                }
                .font(.caption)
            }
        }
        .padding(12)
        .background(.quinary, in: RoundedRectangle(cornerRadius: 10))
        .opacity(enabled ? 1 : 0.7)
    }

    private var candidateLabel: String {
        row.probabilities.first { $0.option.utf8.elementsEqual(row.candidate.utf8) }?.label ?? row.candidate
    }
}
