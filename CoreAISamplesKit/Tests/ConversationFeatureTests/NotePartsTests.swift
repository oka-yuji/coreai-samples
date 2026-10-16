import ConversationFeature
import Testing

struct NotePartsTests {
    @Test(
        "Inline code or a tilde at the start of a line does not hide the following citation",
        arguments: ["`Swift` supports actors [1].", "~2x faster [1]."])
    func checksAfterInlineCode(_ text: String) {
        #expect(throws: NotePartError.missingReference("[1]")) { try NoteParts(first: text) }
    }
    @Test("Citation numbers keep the same destination across note parts")
    func detectsCitationConflicts() throws {
        var parts = try NoteParts(first: "A claim [1].\n\n[1]: https://example.com/a")
        let before = parts
        #expect(throws: NotePartError.conflictingReference("[1]")) {
            try parts.append(part: 2, content: "Another [1](https://example.com/b).")
        }
        #expect(parts == before)
        try parts.append(part: 2, content: "More from [1](https://example.com/a).")
        #expect(parts.parts.count == 2)
        #expect(throws: NotePartError.missingReference("[2]")) {
            try parts.append(part: 3, content: "An unbound claim [2].")
        }
        #expect(parts.parts.count == 2)
    }

    @Test("Citation examples inside code and balanced parentheses in URLs do not create false conflicts")
    func keepsMarkdownExamples() throws {
        var parts = try NoteParts(
            first: "[1](https://example.com/Metal_(API))\n\n```md\n[1]: https://other.example/test")
        try parts.append(
            part: 2,
            content: "```\nMore [1].\n\n[1]: https://example.com/Metal_(API)\n`[1](https://other.example/test)`")
        #expect(parts.parts.count == 2)
    }
}
