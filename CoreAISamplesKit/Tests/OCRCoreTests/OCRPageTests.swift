import OCRCore
import Testing

@Suite("OCR page plan and Markdown")
struct OCRPageTests {
    @Test("Plan one block per layout box in output order and read only table boxes with the table prompt")
    func plannedBlocks() {
        let layout = "<box:053 060 671 091><label:table_caption><up>\n<box:053 124 949 384><label:table><up>"
        let blocks = OCRPage.blocks(layout: layout, width: 1_240, height: 1_754)
        #expect(blocks.map { $0.layout?.label } == ["table_caption", "table"])
        #expect(
            blocks.map(\.box) == [
                OCRBox(x0: 65, y0: 105, x1: 833, y1: 160), OCRBox(x0: 65, y0: 217, x1: 1_177, y1: 674),
            ])
        #expect(blocks.map(\.instruction) == [OCRPrompt.text, OCRPrompt.table])
    }

    @Test(
        "Fall back to the whole page without a layout mark when the layout output has no usable box",
        arguments: [
            "", "Analyze the image layout.", "<box:500 100 500 200><label:text><up>",
            "<box:100 0 101 1000><label:text><up>",
        ])
    func wholePageFallback(layout: String) throws {
        let blocks = OCRPage.blocks(layout: layout, width: 1_000, height: 800)
        let block = try #require(blocks.first)
        #expect(blocks.count == 1)
        #expect(block.layout == nil)
        #expect(block.box == OCRBox(x0: 0, y0: 0, x1: 1_000, y1: 800))
        #expect(block.instruction == OCRPrompt.text)
    }

    @Test("Keep usable boxes in order and record zero, reversed, thin and oversized crops")
    func skippedBoxesAreRecorded() {
        let layout = [
            "<box:0 0 1000 100><label:title><up>",
            "<box:500 100 500 200><label:text><up>",
            "<box:900 100 800 200><label:text><up>",
            "<box:100 0 101 1000><label:text><up>",
            "<box:0 0 40000 40000><label:text><up>",
            "<box:0 200 1000 1000><label:table><up>",
        ].joined(separator: "\n")
        let plan = OCRPage.plan(layout: layout, width: 1_000, height: 800)
        #expect(plan.blocks.map { $0.layout?.label } == ["title", "table"])
        #expect(
            plan.skippedBoxes.map(\.cropBox) == [
                OCRBox(x0: 500, y0: 80, x1: 500, y1: 160),
                OCRBox(x0: 900, y0: 80, x1: 800, y1: 160),
                OCRBox(x0: 100, y0: 0, x1: 101, y1: 800),
                OCRBox(x0: 0, y0: 0, x1: 40_000, y1: 32_000),
            ])
    }

    @Test(
        "Render OTSL as the model card's HTML table, pass HTML through, and return nothing without cells",
        arguments: [
            (
                "<fcel>A<lcel><fcel>B<nl><ucel><xcel><ecel><nl><fcel>C<fcel>D<ucel><nl>",
                "<table><tr><td rowspan=\"2\" colspan=\"2\">A</td><td>B</td></tr><tr><td rowspan=\"2\"></td></tr>"
                    + "<tr><td>C</td><td>D</td></tr></table>"
            ),
            ("<nl><fcel>A<nl><nl><fcel>B<nl>", "<table><tr></tr><tr><td>A</td></tr></table>"),
            (
                "<fcel>a&b<fcel>\"x\" 'y'<nl><fcel><y><lcel><nl>",
                "<table><tr><td>a&amp;b</td><td>&quot;x&quot; &#x27;y&#x27;</td></tr>"
                    + "<tr><td colspan=\"2\">&lt;y&gt;</td></tr></table>"
            ),
            ("<table><tr><td>A</td></tr></table>", "<table><tr><td>A</td></tr></table>"),
            ("| A | B |", ""),
        ])
    func html(otsl: String, html: String) {
        #expect(OCRTable.html(otsl: otsl) == html)
    }

    @Test("Write a title as a heading, a table as HTML, and keep the text of a table without OTSL")
    func markdown() {
        let layout = "<box:0 0 10 10><label:title><box:0 10 10 20><label:table><box:0 20 10 30><label:text>"
        let blocks = OCRPage.blocks(layout: layout, width: 100, height: 100)
        #expect(blocks.map { $0.markdown("A") } == ["# A", "A", "A"])
        #expect(blocks[0].markdown("") == "")
        #expect(blocks[1].markdown("<fcel>A<nl>") == "<table><tr><td>A</td></tr></table>")
    }
}
