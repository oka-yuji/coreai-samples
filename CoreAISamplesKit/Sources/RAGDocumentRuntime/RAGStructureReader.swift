import Foundation
import RAGContext

public enum RAGStructureReader {
    public static let maximumInputBytes = 1_048_576

    public static func read(
        _ document: RAGDocumentSnapshot, normalizeHTML: (String) -> String = RAGHTMLText.plainText
    ) -> RAGDocumentSnapshot {
        var result = document
        guard document.raw.utf8.count <= maximumInputBytes, document.text.utf8.count <= maximumInputBytes else {
            result.structure = .unavailable("structureInputLimit")
            return result
        }
        switch document.format {
        case .markdown:
            result.structure = markdown(document)
        case .html:
            do { result.structure = try html(document, normalize: normalizeHTML) } catch {
                result.structure = .unavailable("htmlParseError")
            }
        case .text, .pdfText, .renderedText:
            result.structure = .unavailable("unsupportedFormat")
        }
        return result
    }
}
