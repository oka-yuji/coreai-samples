#if os(macOS)
    import OCRFeature
    import SwiftUI

    struct OCRResultView: View {
        let document: OCRDocument

        var body: some View {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    if !document.isComplete {
                        Text("The output limit was reached. Some content may be missing.", bundle: .module)
                            .foregroundStyle(.orange)
                    }
                    if document.layout.status == "unparsed" {
                        Text(
                            "The page was read as one region because its layout could not be detected.", bundle: .module
                        )
                        .font(.callout).foregroundStyle(.secondary)
                    }
                    if !document.layout.skippedBoxes.isEmpty {
                        Text("Some regions could not be read.", bundle: .module).foregroundStyle(.orange)
                    }
                    ForEach(document.blocks) { block in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(
                                    LocalizedStringKey(
                                        block.prompt == "table" ? "Table" : block.label == "title" ? "Heading" : "Text"),
                                    bundle: .module
                                )
                                .font(.headline)
                                Text("\(block.index + 1)").foregroundStyle(.secondary)
                            }
                            if let grid = block.grid, !grid.isEmpty {
                                table(grid)
                            } else {
                                Text(verbatim: block.text).textSelection(.enabled).frame(
                                    maxWidth: .infinity, alignment: .leading)
                            }
                        }
                        .padding(12).background(.background, in: RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                    }
                }.padding(.leading, 14).padding(.bottom, 12)
            }
        }

        private func table(_ rows: [[String]]) -> some View {
            ScrollView(.horizontal) {
                Grid(alignment: .topLeading, horizontalSpacing: 0, verticalSpacing: 0) {
                    ForEach(rows.indices, id: \.self) { row in
                        GridRow {
                            ForEach(rows[row].indices, id: \.self) { column in
                                Text(verbatim: rows[row][column]).textSelection(.enabled)
                                    .frame(minWidth: 70, maxWidth: 220, alignment: .leading).padding(8)
                                    .background(row.isMultiple(of: 2) ? Color.secondary.opacity(0.08) : Color.clear)
                                    .border(.quaternary, width: 0.5)
                            }
                        }
                    }
                }
            }
        }
    }
#endif
