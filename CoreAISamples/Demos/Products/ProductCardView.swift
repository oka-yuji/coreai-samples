import Foundation
import ProductCore
import ProductFeature
import SwiftUI

struct ProductCardView: View {
    let card: ProductCard
    let inspect: (ProductField?) -> Void
    let remove: () -> Void
    @Environment(\.locale) private var locale
    private var strings: DemoStrings { DemoStrings(locale: locale) }
    private var memo: ProductMemo { card.memo }
    @ScaledMetric(relativeTo: .body) private var rowHeight = 62
    @ScaledMetric(relativeTo: .title3) private var titleHeight = 44

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: "laptopcomputer").font(.title3)
                    Spacer()
                    Button(action: remove) { Image(systemName: "xmark") }
                        .buttonStyle(.plain).foregroundStyle(.secondary)
                        .accessibilityLabel("Remove product card")
                }
                Text(verbatim: title).font(.title3.weight(.semibold)).lineLimit(2)
                    .frame(height: titleHeight, alignment: .topLeading)
                if let url = memo.source.url {
                    Link(destination: url) { Label(url.host() ?? "Source", systemImage: "arrow.up.right") }
                        .font(.caption).lineLimit(1)
                } else {
                    Label("Pasted description", systemImage: "doc.on.clipboard").font(.caption).foregroundStyle(
                        .secondary)
                }
                if let identity = card.resourceIdentity {
                    Text("Extracted with \(identity)")
                        .font(.caption2).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
            }
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.accentColor.opacity(0.065))
            ForEach(ProductField.allCases.filter { $0 != .product }, id: \.self) { field in
                Divider()
                attribute(field)
            }
            Divider()
            Button("Read source text", systemImage: "doc.text.magnifyingglass") { inspect(nil) }
                .font(.callout).padding(18)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(.background, in: RoundedRectangle(cornerRadius: 16))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(Color.primary.opacity(0.10)))
    }

    private var title: String {
        if !memo.source.title.isEmpty { return memo.source.title }
        return memo.values(for: .product).first?.candidate.text ?? strings.text("Untitled product")
    }

    private func attribute(_ field: ProductField) -> some View {
        let values = memo.values(for: field)
        let distinct = values.reduce(into: [String]()) { result, value in
            if !result.contains(where: { Array($0.utf8) == Array(value.candidate.text.utf8) }) {
                result.append(value.candidate.text)
            }
        }
        return Button {
            inspect(field)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(LocalizedStringKey(field.title)).font(.caption.weight(.medium)).foregroundStyle(.secondary)
                    Spacer()
                    if !values.isEmpty { Image(systemName: "text.quote").font(.caption).foregroundStyle(.tertiary) }
                }
                if values.isEmpty {
                    Text("Not found in this text").font(.callout).foregroundStyle(.secondary)
                } else {
                    Text(verbatim: distinct.prefix(3).joined(separator: " · "))
                        .font(.body.weight(.medium)).lineLimit(2).fixedSize(horizontal: false, vertical: true)
                    if distinct.count > 1 {
                        Text("\(distinct.count) values · check configurations").font(.caption2).foregroundStyle(
                            .secondary)
                    } else {
                        Text("Open the original context").font(.caption2).foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 0)
            }
            .frame(height: rowHeight, alignment: .topLeading)
            .padding(.horizontal, 18).padding(.vertical, 10)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text("Show source for \(strings.text(field.title))"))
        .accessibilityValue(
            values.isEmpty ? Text("Not found in this text") : Text(verbatim: distinct.joined(separator: ", ")))
    }
}

struct ProductInspection: Identifiable {
    let id: Int
    let memo: ProductMemo
    let field: ProductField?
}

struct ProductSourceView: View {
    let inspection: ProductInspection
    @Environment(\.dismiss) private var dismiss
    @Environment(\.locale) private var locale
    private var source: ProductSource { inspection.memo.source }
    private var strings: DemoStrings { DemoStrings(locale: locale) }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text(verbatim: source.title.isEmpty ? strings.text("Product description") : source.title).font(
                        .title3.bold())
                    if let url = source.url { Link(url.absoluteString, destination: url).font(.caption) }
                    Text(
                        "Captured \(source.capturedAt.formatted(.dateTime.year().month().day().hour().minute().locale(locale)))"
                    )
                    .font(.caption).foregroundStyle(.secondary)
                    if let field = inspection.field {
                        Text(LocalizedStringKey(field.title)).font(.headline)
                        let values = inspection.memo.values(for: field)
                        if values.isEmpty {
                            Text("No value was found for this field. The original text is shown below.")
                        }
                        ForEach(Array(values.enumerated()), id: \.offset) { _, value in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(verbatim: value.candidate.text).font(.headline)
                                Text(verbatim: value.context).textSelection(.enabled)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading).padding(14)
                            .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                        }
                        Divider()
                    }
                    Text("Full source text").font(.headline)
                    Text(verbatim: source.text).textSelection(.enabled).lineSpacing(4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(24)
            }
            .demoNavigationTitle("Source and conditions")
            .toolbar { Button("Done") { dismiss() } }
        }
        .frame(idealWidth: 680, idealHeight: 700)
    }
}
