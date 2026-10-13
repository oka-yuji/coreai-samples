import Foundation

struct Catalog: Decodable {
    struct Unit: Decodable {
        let state: String
        let value: String
    }

    struct Localization: Decodable {
        let stringUnit: Unit?
        let variations: [String: [String: Localization]]?
    }

    struct Entry: Decodable {
        let shouldTranslate: Bool?
        let localizations: [String: Localization]?
    }

    let sourceLanguage: String
    let strings: [String: Entry]
}

do {
    guard CommandLine.arguments.count >= 3,
        let bundle = Bundle(url: URL(filePath: CommandLine.arguments[1]))
    else {
        FileHandle.standardError.write(Data("Usage: check-bundle.swift App.bundle Catalog.xcstrings ...\n".utf8))
        exit(1)
    }
    var checked = 0
    var failures: [String] = []
    for path in CommandLine.arguments.dropFirst(2) {
        let url = URL(filePath: path)
        let table = url.deletingPathExtension().lastPathComponent
        let catalog = try JSONDecoder().decode(Catalog.self, from: Data(contentsOf: url))
        var languages = Set([catalog.sourceLanguage])
        for entry in catalog.strings.values {
            languages.formUnion(entry.localizations?.keys.map { $0 } ?? [])
        }
        for language in languages where !bundle.localizations.contains(language) {
            failures.append("Missing compiled language: \(language)")
        }
        for (key, entry) in catalog.strings where entry.shouldTranslate != false {
            for language in languages {
                let localization = entry.localizations?[language]
                let actual = bundle.localizedString(
                    forKey: key, value: nil, table: table,
                    localizations: [Locale.Language(identifier: language)])
                if localization?.variations != nil {
                    if !actual.contains("%#@") { failures.append("Missing plural format: \(language) / \(key)") }
                } else {
                    let expected = localization?.stringUnit?.value ?? key
                    if actual != expected { failures.append("Lookup mismatch: \(language) / \(table) / \(key)") }
                }
                checked += 1
            }
        }
    }
    for failure in failures { print(failure) }
    print("Compiled catalog lookups: \(checked); failures: \(failures.count)")
    exit(failures.isEmpty ? 0 : 1)
} catch {
    FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
    exit(1)
}
