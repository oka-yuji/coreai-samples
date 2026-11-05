#if os(macOS)
    import FeatureStore
    import ReceiptCore
    import ReceiptFeature
    import SwiftUI

    struct ReceiptInspectorView: View {
        let store: Store<ReceiptFeature>
        let itemID: Int
        var scrolling = true
        @Environment(\.accessibilityReduceMotion) private var reduceMotion

        @ViewBuilder var body: some View {
            if scrolling { ScrollView { content } } else { content }
        }

        @ViewBuilder private var content: some View {
            if let item = store.state.items.first(where: { $0.id == itemID }) {
                VStack(alignment: .leading, spacing: 16) {
                    HStack {
                        receiptText("Review fields").font(.headline)
                        Spacer()
                        if item.isConfirmed {
                            Image(systemName: "checkmark.seal.fill").foregroundStyle(.green).accessibilityLabel(
                                receiptText("Confirmed"))
                        }
                    }
                    if let record = item.record {
                        field("Merchant", .merchant)
                        HStack(alignment: .top, spacing: 10) {
                            field("Date · YYYY-MM-DD", .date)
                            field("Total · JPY", .amount)
                        }
                        field("Purchase purpose (optional)", .purpose, vertical: true)
                        Picker(
                            selection: Binding(
                                get: { record.fields.category }, set: { store.send(.category(itemID, $0)) })
                        ) {
                            ForEach(ReceiptCategory.allCases, id: \.self) { category in
                                receiptText(category.title).tag(category)
                            }
                        } label: {
                            receiptText("Category")
                        }.disabled(!store.state.editable)
                        HStack {
                            Button {
                                store.send(.classify(itemID))
                            } label: {
                                receiptText("Suggest again")
                            }.disabled(!store.state.editable)
                            Spacer()
                            Button {
                                store.send(.confirm(itemID))
                            } label: {
                                receiptText(item.isConfirmed ? "Confirmed" : "Confirm receipt")
                            }
                            .buttonStyle(.borderedProminent).tint(.teal)
                            .disabled(!store.state.editable || !record.canConfirm || record.isConfirmed)
                        }
                        if !record.canConfirm && record.issues.isEmpty {
                            receiptText(
                                "Enter a merchant, a valid date and a whole-yen total, then choose a category."
                            ).font(.caption).foregroundStyle(.orange)
                        }
                        ForEach(record.issues, id: \.self) { issue in
                            Label {
                                receiptText(issue.message)
                            } icon: {
                                Image(systemName: "exclamationmark.circle")
                            }
                            .font(.caption).foregroundStyle(.orange)
                        }
                        if let classification = record.classification {
                            Divider()
                            receiptText("JEV category scores").font(.subheadline.bold())
                            if classification.category == .review && classification.topCategory != .review {
                                receiptText(
                                    "No category reached the suggestion threshold. Review the purpose and choose a category."
                                )
                                .font(.caption).foregroundStyle(.orange)
                            }
                            ForEach(Array(ReceiptCategory.allCases.enumerated()), id: \.element) {
                                index, category in
                                HStack(spacing: 8) {
                                    receiptText(category.title).font(.caption).frame(
                                        width: 110, alignment: .leading)
                                    GeometryReader { geometry in
                                        Capsule().fill(.quaternary)
                                            .overlay(alignment: .leading) {
                                                Capsule().fill(
                                                    category == classification.topCategory
                                                        ? Color.orange : Color.secondary.opacity(0.4)
                                                )
                                                .frame(
                                                    width: geometry.size.width
                                                        * Double(classification.probabilities[index]))
                                            }
                                    }.frame(height: 5)
                                    Text(
                                        Double(classification.probabilities[index]),
                                        format: .percent.precision(.fractionLength(1))
                                    )
                                    .font(.caption2.monospacedDigit()).frame(width: 42, alignment: .trailing)
                                }
                            }
                            receiptText(
                                "Scores compare these eight labels. They are not correctness probabilities."
                            )
                            .font(.caption2).foregroundStyle(.secondary)
                        }
                        if let seconds = item.ocrSeconds {
                            HStack {
                                timing("OCR", seconds)
                                Spacer()
                                if let seconds = item.decisionSeconds {
                                    timing("JEV", seconds).help(
                                        receiptText("Time for this JEV call, including any required re-preparation."))
                                }
                            }.padding(10).background(
                                .quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
                        }
                        if let row = store.state.ledger?.rows.first(where: { $0.id == itemID }) {
                            VStack(alignment: .leading, spacing: 5) {
                                receiptText("Export path").font(.caption.bold())
                                Text(verbatim: row.relativePath).font(.system(.caption2, design: .monospaced))
                                    .textSelection(.enabled)
                            }
                        }
                    } else {
                        receiptText("The extracted fields and JEV suggestion appear here.").foregroundStyle(
                            .secondary)
                    }
                    if let failure = item.failure {
                        receiptText(failure).foregroundStyle(.red).font(.callout).textSelection(.enabled)
                    }
                    Divider()
                    receiptText(
                        "Expense labels are suggestions. Confirm business use yourself; this demo does not determine tax treatment."
                    )
                    .font(.caption2).foregroundStyle(.secondary)
                    Button(role: .destructive) {
                        store.send(.remove(itemID))
                    } label: {
                        receiptText("Remove from session")
                    }
                    .disabled(!store.state.editable)
                }.padding(16)
                    .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: item.isConfirmed)
            }
        }

        private func field(_ title: String, _ field: ReceiptField, vertical: Bool = false) -> some View {
            VStack(alignment: .leading, spacing: 6) {
                receiptText(title).font(.caption).foregroundStyle(.secondary)
                TextField(
                    "",
                    text: Binding(
                        get: {
                            guard let values = store.state.items.first(where: { $0.id == itemID })?.record?.fields
                            else { return "" }
                            switch field {
                            case .merchant: return values.merchant
                            case .date: return values.date
                            case .amount: return values.amount
                            case .purpose: return values.purpose
                            }
                        }, set: { store.send(.edit(itemID, field, $0)) }), axis: vertical ? .vertical : .horizontal
                )
                .textFieldStyle(.roundedBorder).lineLimit(vertical ? 2...4 : 1...1)
                .accessibilityLabel(receiptText(title)).disabled(!store.state.editable)
            }
        }

        private func timing(_ label: String, _ seconds: Double) -> some View {
            VStack(alignment: .leading, spacing: 3) {
                Text(verbatim: label).font(.caption2).foregroundStyle(.secondary)
                HStack(alignment: .firstTextBaseline, spacing: 3) {
                    Text(seconds * 1000, format: .number.precision(.fractionLength(0))).monospacedDigit().font(
                        .headline)
                    Text(verbatim: "ms").font(.caption)
                }
            }
        }
    }
#endif
