import FoundationModels

public struct AppendNoteArguments: Generable, Sendable {
    public var noteID: String
    public var part: Int
    public var content: String

    public init(_ value: GeneratedContent) throws {
        noteID = try value.value(String.self, forProperty: "note_id")
        part = try value.value(Int.self, forProperty: "part")
        content = try value.value(String.self, forProperty: "content")
    }

    public static var generationSchema: GenerationSchema {
        GenerationSchema(
            type: Self.self,
            properties: [
                .init(name: "note_id", description: "The note_id returned by write_note.", type: String.self),
                .init(
                    name: "part", description: "The next part number, 2 to 4. Part 1 was saved by write_note.",
                    type: Int.self),
                .init(
                    name: "content", description: "Markdown for this part only. Aim for 1500 characters; at most 2000.",
                    type: String.self),
            ])
    }

    public var generatedContent: GeneratedContent {
        GeneratedContent(properties: ["note_id": noteID, "part": part, "content": content])
    }
}
