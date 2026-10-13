import RecipeCore
import SwiftUI

struct RecipeReportView: View {
    let report: RecipeReport
    @Environment(\.locale) private var locale
    private var strings: DemoStrings { DemoStrings(locale: locale) }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(LocalizedStringKey(report.title), systemImage: summaryIcon)
                .font(.title3.bold()).foregroundStyle(summaryColor)
            ForEach(report.rows) { row in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(verbatim: strings.ingredient(row.ingredient.name)).font(.headline).help(
                            row.ingredient.name)
                        Spacer()
                        Text(LocalizedStringKey(row.decision.title)).font(.caption).foregroundStyle(color(row))
                    }
                    HStack {
                        Text("Recipe: \(row.ingredient.amount)").foregroundStyle(.secondary)
                        Spacer()
                        amountLabel(row.amount).foregroundStyle(row.amount.isShort ? .orange : .secondary)
                    }
                    .font(.callout)
                }
                Divider()
            }
            Label(LocalizedStringKey(equipmentTitle), systemImage: "oven")
                .foregroundStyle(report.equipment == .unavailable ? .orange : .primary)
        }
        .padding(16)
        .background(summaryColor.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
        .textSelection(.enabled)
    }

    @ViewBuilder
    private func amountLabel(_ amount: AmountCheck) -> some View {
        switch amount {
        case .sufficient: Text("Amount checked")
        case .short(let value, let unit): Text("Short by \(strings.number(value)) \(strings.text(unit))")
        case .unknown: Text("Amount not checked")
        case .notRequired: EmptyView()
        }
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
