import Foundation
import RAGContext
import SwiftSoup

extension RAGStructureReader {
    static func html(_ source: RAGDocumentSnapshot, normalize: (String) -> String) throws -> RAGDocumentStructure {
        guard normalize(source.raw).utf8.elementsEqual(source.text.utf8) else {
            return .unavailable("normalizationMismatch")
        }
        let document = try SwiftSoup.parse(source.raw)
        document.outputSettings().prettyPrint(pretty: false)
        var pending: [(Node, Int)] = [(document, 0)]
        var visited = 0
        while let (node, depth) = pending.popLast() {
            visited += 1
            guard depth <= 128, visited <= 16_384 else { return .unavailable("htmlTreeLimit") }
            pending.append(contentsOf: node.getChildNodes().map { ($0, depth + 1) })
        }
        let articles = try document.select("article").array().filter { try !excluded($0) }
        let mains = try document.select("main, [role=main]").array().filter { try !excluded($0) }
        let root: Element
        if articles.count == 1, let article = articles.first {
            root = article
        } else if mains.count == 1, let main = mains.first {
            root = main
        } else if articles.isEmpty && mains.isEmpty, let body = document.body(), try !excluded(body) {
            root = body
        } else {
            return .unavailable("ambiguousContentRoot")
        }
        guard try root.select("nav, aside, footer, template, [hidden], [aria-hidden=true]").isEmpty() else {
            return .unavailable("nonBodyContentInsideRoot")
        }
        for element in try root.select("[style]").array() {
            let style = try element.attr("style").lowercased().filter { !$0.isWhitespace }
            if style.contains("display:none") || style.contains("visibility:hidden") {
                return .unavailable("hiddenContentInsideRoot")
            }
        }
        let nodes = try root.select("h1, h2, h3, h4, h5, h6").array().filter { heading in
            !heading.parents().array().contains {
                ["pre", "code", "head", "script", "style", "noscript"].contains($0.tagName().lowercased())
            }
        }
        guard !nodes.isEmpty else { return .unavailable("noHeadings") }
        guard nodes.count <= 256 else { return .unavailable("headingLimit") }
        let rootText = normalize(try root.html())
        let prefix = "RAGSTRUCT_" + source.version + "_"
        guard !source.raw.contains(prefix), !source.text.contains(prefix) else {
            return .unavailable("markerCollision")
        }
        var labels: [String] = []
        var markers: [String] = []
        for (index, node) in nodes.enumerated() {
            labels.append(normalize(try node.html()))
            let marker = prefix + String(index) + ";"
            markers.append(marker)
            try node.appendText(marker)
        }
        let scopeMarker = prefix + "scope;"
        markers.append(scopeMarker)
        try root.appendText(scopeMarker)
        let annotated = normalize(try document.outerHtml())
        guard let projection = RAGTextProjection.removingMarkers(from: annotated, prefix: prefix, markers: markers),
            projection.text.utf8.elementsEqual(source.text.utf8),
            let scopeEnd = projection.markerEnds[scopeMarker], scopeEnd >= rootText.count
        else { return .unavailable("unmappedHTMLProjection") }
        let content = RAGTextRange(scopeEnd - rootText.count, scopeEnd)
        guard content.text(in: source.text)?.utf8.elementsEqual(rootText.utf8) == true else {
            return .unavailable("unmappedContentRoot")
        }
        var headings: [RAGHeading] = []
        for (index, node) in nodes.enumerated() {
            guard let end = projection.markerEnds[markers[index]], end >= labels[index].count,
                let level = Int(node.tagName().dropFirst())
            else { return .unavailable("unmappedHeading") }
            let range = RAGTextRange(end - labels[index].count, end)
            guard range.text(in: source.text)?.utf8.elementsEqual(labels[index].utf8) == true else {
                return .unavailable("unmappedHeading")
            }
            headings.append(RAGHeading(level: level, title: labels[index], range: range))
        }
        return .make(documentID: source.id, text: source.text, headings: headings, content: content)
    }

    private static func excluded(_ element: Element) throws -> Bool {
        for node in [element] + element.parents().array() {
            if ["head", "nav", "aside", "footer", "template", "script", "style", "noscript", "pre", "code"].contains(
                node.tagName().lowercased())
            {
                return true
            }
            let ariaHidden = try node.attr("aria-hidden").lowercased()
            if node.hasAttr("hidden") || ariaHidden == "true" { return true }
            let style = try node.attr("style").lowercased().filter { !$0.isWhitespace }
            if style.contains("display:none") || style.contains("visibility:hidden") { return true }
        }
        return false
    }
}
