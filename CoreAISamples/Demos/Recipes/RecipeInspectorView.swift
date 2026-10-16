import RecipeCore
import RecipeFeature
import SwiftUI

struct RecipeInspectorView: View {
    let state: RecipeState
    @Environment(\.locale) private var locale
    @State private var expanded = false
    @State private var questionID: String?
    private var strings: DemoStrings { DemoStrings(locale: locale) }

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 16) {
                photoStage
                recipeStage
                decisionStage
                rulesStage
            }
            .padding(.top, 12)
            .textSelection(.enabled)
        } label: {
            Label("Developer details", systemImage: "chart.bar.xaxis")
                .font(.callout.weight(.semibold))
        }
        .padding(14)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 12))
    }

    private var photoStage: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Photo → ingredient suggestions").font(.headline)
            Text(verbatim: "Apple · SystemLanguageModel.default").font(.caption.monospaced())
            Text("Runs on this Mac. Suggestions require a person's confirmation.")
                .font(.caption).foregroundStyle(.secondary)
            if let photo = state.photo {
                LabeledContent("Photo processing", value: seconds(photo.seconds))
                LabeledContent("Suggested ingredients", value: String(photo.ingredients.count))
            } else {
                Text("No photo result yet.").foregroundStyle(.secondary)
            }
        }
        .font(.callout)
    }

    private var recipeStage: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("URL → recipe text").font(.headline)
            Text("Fetches the source page; extracts ingredients and cooking steps.")
                .font(.caption).foregroundStyle(.secondary)
            if let recipe = state.recipe {
                LabeledContent("Recipe ingredients", value: String(recipe.ingredients.count))
                LabeledContent("Cooking steps", value: String(recipe.steps.count))
            } else {
                Text("No recipe loaded yet.").foregroundStyle(.secondary)
            }
        }
        .font(.callout)
    }

    private var decisionStage: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Clef → requirements and equipment").font(.headline)
            Text("The model scores fixed choices. It does not count your stock.")
                .font(.caption).foregroundStyle(.secondary)
            if let report = state.report {
                LabeledContent("Recipe check", value: seconds(report.seconds))
                LabeledContent("Reused document tokens", value: String(report.reusedTokens))
                LabeledContent("Model version") { Text(verbatim: report.modelVersion) }
                Text(
                    "Check time includes document preparation and result mapping; excludes opening the selected bundle and queueing."
                )
                .font(.caption2).foregroundStyle(.secondary)
                if let diagnostics = report.diagnostics {
                    LabeledContent("Result model") { Text(verbatim: diagnostics.model) }
                    LabeledContent("Input tokens", value: String(diagnostics.inputTokens))
                    LabeledContent("Questions", value: String(diagnostics.questions.count))
                    Text("Input tokens cover the recipe and all questions. Reuse is zero on the first check.")
                        .font(.caption2).foregroundStyle(.secondary)
                    questionDetail(diagnostics, report: report)
                } else {
                    Text("Choice scores were not recorded for this result.").foregroundStyle(.secondary)
                }
            } else {
                Text("Check a recipe to see the returned scores.").foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .monospacedDigit()
    }

    @ViewBuilder
    private func questionDetail(_ diagnostics: RecipeDiagnostics, report: RecipeReport) -> some View {
        let question = diagnostics.questions.first { $0.id == questionID } ?? diagnostics.questions.first
        if let question {
            Divider()
            Picker(
                "Inspect question",
                selection: Binding(get: { question.id }, set: { questionID = $0 })
            ) {
                ForEach(diagnostics.questions) { item in
                    Text(verbatim: questionTitle(item, report: report)).tag(item.id)
                }
            }
            .pickerStyle(.menu)
            Text(verbatim: question.id).font(.caption2.monospaced()).foregroundStyle(.secondary)
            ForEach(question.options) { option in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(LocalizedStringKey(option.option))
                        if option.option == question.selected {
                            Image(systemName: "checkmark.circle.fill").foregroundStyle(.tint)
                                .accessibilityLabel(Text("Selected choice"))
                        }
                        Spacer()
                        Text(verbatim: strings.number(Double(option.probability), fractionDigits: 4...4))
                    }
                    ProgressView(value: Double(option.probability), total: 1)
                        .accessibilityLabel(Text(LocalizedStringKey(option.option)))
                    Text("Logit: \(strings.number(Double(option.logit), fractionDigits: 3...3))")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                .padding(.vertical, 3)
            }
            Text("Probabilities are normalized across these choices and are not calibrated confidence.")
                .font(.caption).foregroundStyle(.secondary)
            if let instructions = question.instructions {
                DisclosureGroup("Model question (original)") {
                    Text(verbatim: instructions).font(.caption.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 4)
                }
            }
        }
    }

    private var rulesStage: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Code → stock and quantity checks").font(.headline)
            Text(
                "Confirmed names and amounts are matched in code. Shortages are calculated, not generated by the model."
            )
            Text("Review threshold: \(strings.number(Double(RecipeReport.reviewThreshold), fractionDigits: 2...2))")
            Text(
                "A score below this threshold, an uncertain requirement, or unknown equipment, stock, or amount needs review."
            )
            .foregroundStyle(.secondary)
        }
        .font(.caption)
    }

    private func questionTitle(_ question: RecipeQuestionDiagnostic, report: RecipeReport) -> String {
        if question.id == "equipment" { return strings.text("Kitchen equipment") }
        if let row = report.rows.first(where: { "ingredient_\($0.id)" == question.id }) {
            return strings.ingredient(row.ingredient.name)
        }
        return question.id
    }

    private func seconds(_ value: Double) -> String {
        strings.number(value, fractionDigits: 2...2) + " s"
    }
}
