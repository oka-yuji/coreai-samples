import Foundation

public enum RecipeURL {
    public static func validate(_ value: String) throws -> URL {
        guard let parts = URLComponents(string: value), parts.scheme?.lowercased() == "https",
            parts.host?.lowercased() == "www.kikkoman.co.jp", parts.user == nil, parts.password == nil,
            parts.port == nil || parts.port == 443, parts.query == nil,
            parts.percentEncodedPath.range(
                of: #"^/homecook/search/recipe/[0-9]{8}/$"#, options: .regularExpression) != nil,
            let url = parts.url
        else { throw RecipeError.invalid("Use a Kikkoman recipe URL, such as one of the five examples.") }
        return url
    }
}

public enum RecipeParser {
    public static func parse(html: String, source: String) throws -> Recipe {
        _ = try RecipeURL.validate(source)
        guard html.utf8.count <= 2_000_000 else { throw RecipeError.invalid("The recipe page is too large.") }
        let scriptPattern = #"<script\b[^>]*\btype\s*=\s*["']application/ld\+json["'][^>]*>([\s\S]*?)</script\s*>"#
        let expression = try NSRegularExpression(pattern: scriptPattern, options: .caseInsensitive)
        let range = NSRange(html.startIndex..., in: html)
        var recipes: [[String: Any]] = []
        for match in expression.matches(in: html, range: range) {
            guard let bodyRange = Range(match.range(at: 1), in: html) else { continue }
            let body = String(html[bodyRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let value = try? JSONSerialization.jsonObject(with: Data(body.utf8)) else { continue }
            collect(value, into: &recipes)
        }
        guard recipes.count == 1, let value = recipes.first,
            let title = value["name"] as? String, !clean(title).isEmpty, title.count <= 500,
            let lines = value["recipeIngredient"] as? [String], !lines.isEmpty, lines.count <= 14,
            let stepsValue = value["recipeInstructions"]
        else { throw RecipeError.invalid("This page must contain one recipe with 1 to 14 ingredients and a method.") }
        let steps = try instructions(stepsValue).map(clean)
        guard !steps.isEmpty, steps.allSatisfy({ !$0.isEmpty }), steps.count <= 60,
            lines.allSatisfy({ !clean($0).isEmpty && $0.count <= 300 }),
            steps.reduce(0, { $0 + $1.utf8.count }) <= 40_000
        else { throw RecipeError.invalid("The recipe ingredients or method could not be read completely.") }
        let ingredients = lines.enumerated().map { index, line in
            let line = clean(line)
            for position in line.indices where line[position] == " " {
                let suffix = String(line[line.index(after: position)...])
                if suffix.range(
                    of: #"^(?:\p{Nd}|\u5927\u3055\u3058|\u5C0F\u3055\u3058|\u9069|\u5C11|to taste|a pinch)"#,
                    options: .regularExpression) != nil
                {
                    return RecipeIngredient(id: index, name: String(line[..<position]), amount: suffix)
                }
            }
            return RecipeIngredient(id: index, name: line, amount: "")
        }
        let yield: String
        if let string = value["recipeYield"] as? String {
            yield = string
        } else if let strings = value["recipeYield"] as? [String] {
            yield = strings.joined(separator: " / ")
        } else {
            yield = "Not stated"
        }
        return Recipe(
            title: clean(title), source: source, servings: yield, duration: value["totalTime"] as? String ?? "",
            ingredients: ingredients, steps: steps)
    }

    private static func collect(_ value: Any, into recipes: inout [[String: Any]]) {
        if let list = value as? [Any] {
            for item in list { collect(item, into: &recipes) }
        } else if let object = value as? [String: Any] {
            let types = (object["@type"] as? [String]) ?? [object["@type"] as? String ?? ""]
            if types.contains("Recipe") { recipes.append(object) }
            if let graph = object["@graph"] { collect(graph, into: &recipes) }
        }
    }

    private static func instructions(_ value: Any) throws -> [String] {
        if let text = value as? String { return [text] }
        if let list = value as? [Any], !list.isEmpty { return try list.flatMap(instructions) }
        if let object = value as? [String: Any] {
            if let children = object["itemListElement"] {
                let steps = try instructions(children)
                if let text = object["text"] as? String, !clean(text).isEmpty { return [text] + steps }
                return steps
            }
            if let text = object["text"] as? String { return [text] }
        }
        throw RecipeError.invalid("A recipe step has no readable instruction text.")
    }

    private static func clean(_ value: String) -> String {
        value.replacingOccurrences(of: #"<[^>]*>"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
