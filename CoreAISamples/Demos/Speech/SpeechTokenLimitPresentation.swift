import Foundation

enum SpeechTokenLimitPresentation {
    static let key = "This text contains %lld tokens. Shorten it to %lld tokens or fewer."

    static func message(count: Int, limit: Int, locale: Locale, bundle: Bundle = .main) -> String {
        let format = bundle.localizedString(
            forKey: key, value: key, table: nil, localizations: [Locale.Language(identifier: locale.identifier)])
        return String(format: format, locale: locale, Int64(count), Int64(limit))
    }
}
