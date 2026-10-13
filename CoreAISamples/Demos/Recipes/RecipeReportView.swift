import RecipeCore
import SwiftUI

struct RecipeReportView: View {
    let report: RecipeReport

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(report.title, systemImage: summaryIcon)
                .font(.title3.bold()).foregroundStyle(summaryColor)
            ForEach(report.rows) { row in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(row.ingredient.name).font(.headline)
                        Spacer()
                        Text(row.decision.title).font(.caption).foregroundStyle(color(row))
                    }
                    HStack {
                        Text("Recipe: " + row.ingredient.amount).foregroundStyle(.secondary)
                        Spacer()
                        Text(row.amount.title).foregroundStyle(row.amount.isShort ? .orange : .secondary)
                    }
                    .font(.callout)
                    Text(
                        "Requirement: " + row.requirement.rawValue + " · Model score "
                            + row.probability.formatted(.number.precision(.fractionLength(2)))
                    )
                    .font(.caption2).foregroundStyle(.secondary)
                }
                Divider()
            }
            Label(equipmentTitle, systemImage: "oven")
                .foregroundStyle(report.equipment == .unavailable ? .orange : .primary)
            Text("Model scores are uncalibrated. Confirm unclear requirements in the original recipe.")
                .font(.caption).foregroundStyle(.secondary)
            Text(
                report.seconds.formatted(.number.precision(.fractionLength(2))) + " s · "
                    + String(report.reusedTokens) + " reused document tokens"
            )
            .font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
        }
        .padding(16)
        .background(summaryColor.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
        .textSelection(.enabled)
    }

    private var summaryIcon: String {
        report.hasShortage || report.equipment == .unavailable
            ? "basket" : report.needsReview ? "questionmark.circle" : "checkmark.circle"
    }

    private var summaryColor: Color {
        report.hasShortage || report.equipment == .unavailable ? .orange : report.needsReview ? .secondary : .green
    }

    private var equipmentTitle: String {
        switch report.equipment {
        case .available: "The stated equipment is available"
        case .unavailable: "Required equipment is missing"
        case .unknown: "Check the required equipment"
        }
    }

    private func color(_ row: RecipeCheckRow) -> Color {
        if row.amount.isShort || row.decision == .missing { return .orange }
        if row.decision == .have && row.amount == .sufficient { return .green }
        return .secondary
    }
}
