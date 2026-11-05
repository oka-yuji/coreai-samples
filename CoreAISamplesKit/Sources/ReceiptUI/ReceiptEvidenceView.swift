#if os(macOS)
    import AppKit
    import ReceiptFeature
    import SwiftUI

    struct ReceiptEvidenceView: View {
        let item: ReceiptItem
        @State private var showingText = false

        var body: some View {
            VStack(spacing: 12) {
                HStack {
                    receiptText("Original receipt").font(.headline)
                    Spacer()
                    Picker(selection: $showingText) {
                        receiptText("Image").tag(false)
                        receiptText("OCR text").tag(true)
                    } label: {
                        receiptText("Receipt evidence")
                    }.pickerStyle(.segmented).labelsHidden().frame(width: 170)
                }
                if showingText {
                    ScrollView {
                        Text(verbatim: item.record?.text ?? "").font(.system(.callout, design: .monospaced))
                            .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(16)
                    }.background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
                } else if !item.previews.isEmpty {
                    ScrollView {
                        VStack(spacing: 16) {
                            ForEach(Array(item.previews.enumerated()), id: \.offset) { index, data in
                                if let image = NSImage(data: data) {
                                    Image(nsImage: image).resizable().scaledToFit().frame(maxWidth: 580)
                                        .clipShape(RoundedRectangle(cornerRadius: 3))
                                        .shadow(color: .black.opacity(0.12), radius: 5, y: 2)
                                        .frame(maxWidth: .infinity)
                                        .accessibilityLabel(Text("Receipt page \(index + 1)", bundle: .module))
                                }
                            }
                        }.padding(18)
                    }.background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "doc.viewfinder").font(.largeTitle)
                        receiptText("The original preview appears after reading.")
                    }.foregroundStyle(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                Text(verbatim: item.input.url.lastPathComponent).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    .truncationMode(.middle)
            }.padding(15)
        }
    }
#endif
