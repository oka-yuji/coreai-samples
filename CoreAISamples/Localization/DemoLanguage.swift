import Foundation
import SwiftUI

enum DemoLanguage: String, CaseIterable {
    case english = "en"
    case japanese = "ja"

    static let preferenceKey = "demoInterfaceLanguage"

    static var initialCode: String {
        Locale.preferredLanguages.first?.hasPrefix("ja") == true ? "ja" : "en"
    }

    var locale: Locale { Locale(identifier: rawValue) }
    var segment: String { self == .english ? "EN" : "JP" }
}

extension View {
    func demoNavigationTitle(_ key: String) -> some View {
        modifier(DemoNavigationTitle(key: key))
    }
}

private struct DemoNavigationTitle: ViewModifier {
    let key: String
    @Environment(\.locale) private var locale

    func body(content: Content) -> some View {
        content.navigationTitle(Text(verbatim: DemoStrings(locale: locale).text(key)))
    }
}

struct DemoLanguagePicker: View {
    @Binding var code: String

    var body: some View {
        Picker("Interface language", selection: $code) {
            ForEach(DemoLanguage.allCases, id: \.rawValue) { language in
                Text(verbatim: language.segment).tag(language.rawValue)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(width: 112)
        .help("Switch between English and Japanese")
        .accessibilityLabel("Interface language")
    }
}

struct DemoStrings {
    let locale: Locale

    func text(_ key: String) -> String {
        Bundle.main.localizedString(
            forKey: key, value: key, table: nil,
            localizations: [Locale.Language(identifier: locale.identifier)])
    }

    func ingredient(_ name: String) -> String {
        let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return Bundle.main.localizedString(
            forKey: key, value: name, table: "Ingredients",
            localizations: [Locale.Language(identifier: locale.identifier)])
    }

    func number(_ value: Double, fractionDigits: ClosedRange<Int> = 0...2) -> String {
        value.formatted(.number.precision(.fractionLength(fractionDigits)).locale(locale))
    }
}

struct RuntimeMessageText: View {
    let message: String
    @Environment(\.locale) private var locale

    init(_ message: String) {
        self.message = message
    }

    var body: some View {
        let translated = DemoStrings(locale: locale).text(message)
        if translated == message && locale.language.languageCode?.identifier == "ja" && !message.isEmpty {
            Text("Details (original): \(message)")
        } else {
            Text(verbatim: translated)
        }
    }
}
