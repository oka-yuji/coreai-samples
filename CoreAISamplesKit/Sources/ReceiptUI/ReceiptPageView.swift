#if os(macOS)
    import FeatureStore
    import Foundation
    import ReceiptCore
    import ReceiptFeature
    import SwiftUI
    import UniformTypeIdentifiers

    public struct ReceiptPageView: View {
        private let store: Store<ReceiptFeature>
        @State private var choosingFiles = false
        @State private var hovering = false
        @Environment(\.accessibilityReduceMotion) private var reduceMotion

        public init(store: Store<ReceiptFeature>) { self.store = store }

        public var body: some View {
            VStack(spacing: 0) {
                header
                Divider()
                controls
                progress
                Divider()
                if store.state.items.isEmpty {
                    welcome
                } else {
                    GeometryReader { geometry in
                        HSplitView {
                            ReceiptQueueView(store: store).frame(minWidth: 170, idealWidth: 210, maxWidth: 270)
                            if let item = store.state.selected {
                                if geometry.size.width >= 920 {
                                    ReceiptEvidenceView(item: item).frame(minWidth: 250, maxWidth: .infinity)
                                    ReceiptInspectorView(store: store, itemID: item.id).frame(
                                        minWidth: 290, idealWidth: 320, maxWidth: 370)
                                } else {
                                    ScrollView {
                                        VStack(spacing: 18) {
                                            ReceiptEvidenceView(item: item).frame(height: 340)
                                            ReceiptInspectorView(store: store, itemID: item.id, scrolling: false)
                                        }
                                    }.frame(minWidth: 320, maxWidth: .infinity)
                                }
                            }
                        }
                    }
                }
                Divider()
                categories
                footer
            }
            .background(.background)
            .overlay {
                if hovering {
                    RoundedRectangle(cornerRadius: 12).stroke(.orange, lineWidth: 3).padding(4).allowsHitTesting(false)
                }
            }
            .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: store.state.phase)
            .fileImporter(
                isPresented: $choosingFiles, allowedContentTypes: [.jpeg, .png, .heic, .heif, .tiff, .pdf],
                allowsMultipleSelection: true
            ) { result in
                switch result {
                case .success(let urls): store.send(.add(urls.map { ReceiptInput(url: $0) }))
                case .failure(let error):
                    if (error as? CocoaError)?.code != .userCancelled {
                        store.send(.uiError(error.localizedDescription))
                    }
                }
            }
            .dropDestination(for: URL.self) { urls, _ in
                guard store.state.editable, !urls.isEmpty, urls.allSatisfy(\.isFileURL) else { return false }
                store.send(.add(urls.map { ReceiptInput(url: $0) }))
                return true
            } isTargeted: {
                hovering = $0
            }
            .background {
                if let selection = store.state.exportSelection {
                    ReceiptExportPicker(selection: selection, send: store.send).id(selection.id)
                }
            }
        }

        private var header: some View {
            HStack(alignment: .center, spacing: 24) {
                VStack(alignment: .leading, spacing: 5) {
                    HStack(spacing: 9) {
                        Image(systemName: "receipt").foregroundStyle(.orange)
                        Text(verbatim: "Receipt Desk").font(.title.bold())
                    }
                    receiptText("From receipts to a reviewed expense ledger.").foregroundStyle(.secondary)
                }
                Spacer(minLength: 12)
                VStack(alignment: .trailing, spacing: 3) {
                    receiptText("Confirmed total · JPY").font(.caption).foregroundStyle(.secondary)
                    if let total = store.state.ledger?.totalYen {
                        Text(total, format: .number).font(.system(size: 30, weight: .semibold, design: .rounded))
                            .monospacedDigit()
                    } else {
                        Text(verbatim: store.state.confirmedCount == 0 ? "0" : "—").font(
                            .system(size: 30, weight: .semibold, design: .rounded))
                    }
                    Text(
                        "\(store.state.confirmedCount) confirmed / \(store.state.items.count) receipts", bundle: .module
                    )
                    .font(.caption).foregroundStyle(.secondary)
                }
            }.padding(.horizontal, 22).padding(.vertical, 17)
        }

        private var controls: some View {
            HStack(spacing: 10) {
                Button {
                    choosingFiles = true
                } label: {
                    Label {
                        receiptText("Add & Process…")
                    } icon: {
                        Image(systemName: "plus")
                    }
                }
                .disabled(!store.state.editable).keyboardShortcut("o")
                if store.state.items.isEmpty {
                    Button {
                        store.send(.samples)
                    } label: {
                        receiptText("Run 4 sample receipts")
                    }
                    .disabled(!store.state.editable)
                } else {
                    Button {
                        store.send(.process)
                    } label: {
                        receiptText("Process remaining")
                    }.disabled(!store.state.canProcess)
                }
                if store.state.busy {
                    Button {
                        store.send(.stop)
                    } label: {
                        Label {
                            receiptText("Stop")
                        } icon: {
                            Image(systemName: "stop.fill")
                        }
                    }
                    .disabled([.stopping, .closing, .closed].contains(store.state.phase))
                }
                Spacer(minLength: 8)
                Button {
                    store.send(.confirmTogether)
                } label: {
                    receiptText("Confirm eligible")
                }.disabled(!store.state.canConfirmTogether)
                    .help(
                        receiptText(
                            "Confirm complete receipts with no OCR warnings. Other receipts need individual review."))
                Button {
                    store.send(.requestExport)
                } label: {
                    Label {
                        receiptText("Export…")
                    } icon: {
                        Image(systemName: "folder.badge.plus")
                    }
                }
                .buttonStyle(.borderedProminent).disabled(!store.state.canExport)
            }.controlSize(.regular).padding(.horizontal, 22).padding(.vertical, 11)
        }

        private var progress: some View {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 9) {
                    if store.state.busy {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "checkmark.circle").foregroundStyle(.secondary)
                    }
                    receiptText(store.state.phase.label).font(.callout)
                    if let active = store.state.active,
                        let item = store.state.items.first(where: { $0.id == active.item })
                    {
                        Text(verbatim: item.input.url.lastPathComponent).font(.callout).foregroundStyle(.secondary)
                            .lineLimit(1).truncationMode(.middle)
                    }
                    Spacer()
                    if store.state.batchTotal > 0 {
                        Text("\(store.state.batchDone) / \(store.state.batchTotal) processed", bundle: .module)
                            .monospacedDigit().font(.caption)
                    }
                }
                if store.state.busy && store.state.batchTotal > 0 && store.state.phase != .preparing {
                    ProgressView(value: Double(store.state.batchDone), total: Double(store.state.batchTotal)).tint(
                        .orange
                    )
                    .accessibilityLabel(receiptText("Receipt processing"))
                }
                if let message = store.state.message ?? store.state.ledgerProblem {
                    receiptText(message).font(.callout).foregroundStyle(.red).textSelection(.enabled)
                }
                if let saved = store.state.saved {
                    VStack(alignment: .leading, spacing: 4) {
                        Label {
                            Text("Saved \(saved.count) receipts", bundle: .module)
                        } icon: {
                            Image(systemName: "checkmark.seal.fill")
                        }.foregroundStyle(.green)
                        Text(verbatim: saved.directory.path).font(.caption).textSelection(.enabled)
                    }
                }
            }.padding(.horizontal, 22).padding(.bottom, 14)
        }

        private var welcome: some View {
            VStack(spacing: 22) {
                Image(systemName: "doc.text.image").font(.system(size: 58, weight: .light)).foregroundStyle(.orange)
                receiptText("Drop receipts. Watch them become a ledger.").font(.title2.bold())
                receiptText("JEV suggests categories on this Mac. You check the fields and decide what to save.")
                    .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 470)
                HStack(spacing: 28) {
                    welcomeStep("doc.viewfinder", "Read text")
                    Image(systemName: "arrow.right").foregroundStyle(.tertiary)
                    welcomeStep("tag", "Suggest a category")
                    Image(systemName: "arrow.right").foregroundStyle(.tertiary)
                    welcomeStep("folder", "Save reviewed copies")
                }.padding(.top, 6)
                receiptText("JPEG, PNG, HEIC, TIFF or PDF · JPY · Up to 20 files, 20 MiB each")
                    .font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, maxHeight: .infinity).padding(26)
        }

        private func welcomeStep(_ symbol: String, _ title: String) -> some View {
            VStack(spacing: 10) {
                Image(systemName: symbol).font(.title2).foregroundStyle(.secondary)
                receiptText(title).font(.callout)
            }
        }

        private var categories: some View {
            ScrollView(.horizontal) {
                HStack(spacing: 12) {
                    ForEach(ReceiptCategory.allCases, id: \.self) { category in
                        let count = store.state.items.filter { $0.record?.fields.category == category }.count
                        HStack(spacing: 8) {
                            Image(systemName: category.symbol).foregroundStyle(
                                category == .review ? .orange : .secondary)
                            VStack(alignment: .leading, spacing: 3) {
                                receiptText(category.title).font(.caption).lineLimit(1)
                                Text(count, format: .number).font(.title3.monospacedDigit().bold())
                                    .contentTransition(.numericText())
                            }
                        }
                        .frame(minWidth: 108, alignment: .leading).padding(10)
                        .background(.quaternary.opacity(count > 0 ? 0.7 : 0.25), in: RoundedRectangle(cornerRadius: 8))
                        .animation(reduceMotion ? nil : .easeInOut(duration: 0.25), value: count)
                        .accessibilityElement(children: .combine)
                    }
                }.padding(.horizontal, 20).padding(.vertical, 12)
            }.scrollIndicators(.hidden)
        }

        private var footer: some View {
            HStack(spacing: 6) {
                Image(systemName: "lock.shield")
                receiptText("On this Mac · Vision OCR + JEV on Core AI")
                Spacer()
                receiptText("Export before leaving. Drafts belong to this session.")
            }.font(.caption2).foregroundStyle(.secondary).padding(.horizontal, 22).padding(.bottom, 12)
        }
    }
#endif
