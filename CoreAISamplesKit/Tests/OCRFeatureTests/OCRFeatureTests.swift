import FeatureStore
import Foundation
import OCRCore
import OCRFeature
import Testing

struct OCRFeatureTests {
    @Test("A selected image progresses through load and reading into one result and both export formats")
    func completedPage() throws {
        var state = OCRState(bundle: URL(filePath: "/fixture/model"))
        #expect(OCRFeature.reduce(&state, .read).isEmpty)
        #expect(OCRFeature.reduce(&state, .appeared).isEmpty)
        let url = URL(filePath: "/fixture/page.png")
        #expect(OCRFeature.reduce(&state, .selectImage(url)) == [.run(.select(1, url), id: .operation)])
        let input = try input()
        #expect(OCRFeature.reduce(&state, .selected(1, input)).isEmpty)
        #expect(state.canRead)
        #expect(OCRFeature.reduce(&state, .read) == [.run(.load(2, state.bundle), id: .operation)])
        #expect(state.phase == .loading)
        #expect(OCRFeature.reduce(&state, .loaded(2)) == [.run(.read(2, input), id: .operation)])
        let document = try document()
        #expect(OCRFeature.reduce(&state, .completed(2, document)).isEmpty)
        #expect(state.phase == .complete && state.document == document && document.isComplete)
        #expect(OCRFeature.reduce(&state, .exportRequested(.markdown)).isEmpty)
        let markdown = try #require(state.export)
        #expect(markdown.data == Data((document.markdown + "\n").utf8))
        #expect(markdown.name == "page" && markdown.session == 2)
        #expect(OCRFeature.reduce(&state, .exportFinished(id: markdown.id, session: 1, error: nil)).isEmpty)
        #expect(state.export == markdown)
        _ = OCRFeature.reduce(&state, .exportFinished(id: markdown.id, session: 2, error: nil))
        _ = OCRFeature.reduce(&state, .exportRequested(.json))
        let exported = try #require(state.export)
        let json = try #require(JSONSerialization.jsonObject(with: exported.data) as? [String: Any])
        #expect(json["schema"] as? String == "teleocr-ocr-result/1")
        #expect(json["image_sha256"] as? String == input.sha256)
        let blocks = try #require(json["blocks"] as? [[String: Any]])
        #expect(blocks.first?["prompt"] as? String == "table")
        #expect((blocks.first?["cells"] as? [[String: Any]])?.count == 4)
        #expect(document.markdown.contains("<td>A&amp;</td>"))
    }

    @Test("Stopping invalidates old results and waits for release before another image can start")
    func stopping() throws {
        let input = try input()
        var state = OCRState(bundle: URL(filePath: "/fixture/model"), input: input)
        _ = OCRFeature.reduce(&state, .appeared)
        _ = OCRFeature.reduce(&state, .read)
        _ = OCRFeature.reduce(&state, .loaded(1))
        #expect(OCRFeature.reduce(&state, .stop) == [.cancelAll, .run(.release(2), id: .stop)])
        #expect(state.phase == .stopping && !state.canRead)
        #expect(OCRFeature.reduce(&state, .completed(1, try document())).isEmpty)
        #expect(OCRFeature.reduce(&state, .failed(1, "late")).isEmpty)
        #expect(OCRFeature.reduce(&state, .stopped(1)).isEmpty)
        #expect(OCRFeature.reduce(&state, .read).isEmpty)
        #expect(state.phase == .stopping && state.document == nil)
        _ = OCRFeature.reduce(&state, .stopped(2))
        #expect(state.phase == .cancelled && state.canRead)
        #expect(OCRFeature.reduce(&state, .read) == [.run(.load(3, state.bundle), id: .operation)])
        #expect(OCRFeature.reduce(&state, .stopped(2)).isEmpty)
        #expect(state.phase == .loading)
    }

    @Test("Failures can retry while an independent cancellation still waits for release")
    func failureAndCancellation() throws {
        var state = OCRState(bundle: URL(filePath: "/fixture/model"), input: try input())
        _ = OCRFeature.reduce(&state, .appeared)
        _ = OCRFeature.reduce(&state, .read)
        _ = OCRFeature.reduce(&state, .failed(1, "failure"))
        #expect(state.phase == .failed && state.message == "failure" && state.canRead)
        _ = OCRFeature.reduce(&state, .read)
        #expect(state.message == nil)
        #expect(OCRFeature.reduce(&state, .cancelled(2)) == [.cancelAll, .run(.release(3), id: .stop)])
        #expect(state.phase == .stopping)
        _ = OCRFeature.reduce(&state, .stopped(3))
        #expect(state.phase == .cancelled)
    }

    @Test("Ending cancels work, invalidates exports and ignores late input until release completes")
    func ended() throws {
        var state = OCRState(bundle: URL(filePath: "/fixture/model"), input: try input())
        _ = OCRFeature.reduce(&state, .appeared)
        _ = OCRFeature.reduce(&state, .read)
        #expect(OCRFeature.reduce(&state, .ended) == [.cancelAll, .run(.release(2), id: .stop)])
        #expect(!state.presented)
        #expect(OCRFeature.reduce(&state, .selected(1, try input())).isEmpty)
        #expect(OCRFeature.reduce(&state, .read).isEmpty)
        _ = OCRFeature.reduce(&state, .stopped(2))
        #expect(state.phase == .idle && !state.canRead)
    }

    @Test("Truncated results retain their finish reason and invalid JSON values do not create an export")
    func incompleteResult() throws {
        let limited = try document(finish: "max_new_tokens")
        #expect(!limited.isComplete)
        let json = try #require(JSONSerialization.jsonObject(with: limited.json()) as? [String: Any])
        let layout = try #require(json["layout"] as? [String: Any])
        #expect(layout["finished_by"] as? String == "max_new_tokens")
        var state = OCRState(bundle: URL(filePath: "/fixture/model"), input: try input())
        _ = OCRFeature.reduce(&state, .appeared)
        _ = OCRFeature.reduce(&state, .read)
        _ = OCRFeature.reduce(&state, .loaded(1))
        _ = OCRFeature.reduce(&state, .completed(1, try document(seconds: .nan)))
        _ = OCRFeature.reduce(&state, .exportRequested(.json))
        #expect(state.export == nil && state.message != nil)
    }

    private func input() throws -> OCRInput {
        OCRInput(
            name: "page.png",
            image: try OCRImage(width: 28, height: 28, rgb: [UInt8](repeating: 0, count: 28 * 28 * 3)),
            preview: Data(), sha256: String(repeating: "a", count: 64))
    }

    private func document(finish: String = "eos", seconds: Double = 1) throws -> OCRDocument {
        let input = try input()
        let source = "<box:000 000 999 999><label:table>"
        let block = try #require(OCRLayout.blocks(in: source, width: 28, height: 28).first)
        let timing = OCRDocument.Seconds(vision: 0.1, firstToken: 0.2, total: seconds)
        let tokens = OCRDocument.Tokens(prompt: 4, image: 1, output: 8)
        return OCRDocument(
            bundle: "teleocr-fp16", input: input,
            layout: .init(
                parsed: true, text: source, skippedBoxes: [], seconds: timing, tokens: tokens, finishedBy: finish),
            blocks: [
                .init(
                    index: 0, layout: block, box: block.cropBox, text: "<fcel>A&<fcel>B<nl><fcel>1<fcel>2<nl>",
                    seconds: timing, tokens: tokens, finishedBy: "eos")
            ], generateS: seconds, loadS: 0.5)
    }
}
