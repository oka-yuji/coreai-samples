#if os(macOS)
    import FeatureStore
    import ReceiptCore
    import ReceiptFeature
    import SwiftUI

    struct ReceiptQueueView: View {
        let store: Store<ReceiptFeature>
        @Environment(\.accessibilityReduceMotion) private var reduceMotion

        var body: some View {
            VStack(alignment: .leading, spacing: 0) {
                receiptText("Receipts").font(.headline).padding(15)
                List(
                    selection: Binding(
                        get: { store.state.selection }, set: { if let id = $0 { store.send(.select(id)) } })
                ) {
                    ForEach(store.state.items) { item in
                        HStack(alignment: .top, spacing: 9) {
                            Image(systemName: symbol(item)).foregroundStyle(color(item)).frame(width: 20)
                            VStack(alignment: .leading, spacing: 5) {
                                if item.record == nil, let title = item.input.title {
                                    receiptText(title).font(.callout.bold())
                                } else {
                                    Text(verbatim: item.title).font(.callout.bold()).lineLimit(2)
                                }
                                receiptText(status(item)).font(.caption).foregroundStyle(.secondary)
                                if let amount = item.record?.fields.parsedAmount {
                                    Text("JPY \(amount.yen)", bundle: .module).font(.caption.monospacedDigit())
                                }
                            }
                        }.padding(.vertical, 7).tag(item.id)
                    }
                }.listStyle(.sidebar)
            }
            .animation(reduceMotion ? nil : .easeInOut(duration: 0.18), value: store.state.readCount)
        }

        private func symbol(_ item: ReceiptItem) -> String {
            if store.state.active?.item == item.id { return "circle.dotted.circle" }
            if item.failure != nil { return "exclamationmark.triangle" }
            if item.isConfirmed { return "checkmark.seal.fill" }
            return item.record?.fields.category.symbol ?? "doc"
        }

        private func color(_ item: ReceiptItem) -> Color {
            if item.failure != nil { return .red }
            if item.isConfirmed { return .green }
            if store.state.active?.item == item.id || item.record?.fields.category == .review { return .orange }
            return .secondary
        }

        private func status(_ item: ReceiptItem) -> String {
            if store.state.active?.item == item.id { return store.state.phase.label }
            if item.failure != nil { return "Needs attention" }
            if item.isConfirmed { return "Confirmed" }
            if let record = item.record { return record.fields.category.title }
            return "Waiting to process"
        }
    }
#endif
