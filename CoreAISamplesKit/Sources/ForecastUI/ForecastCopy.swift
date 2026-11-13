import ForecastFeature
import Foundation

enum ForecastCopy {
    static func text(_ key: String, locale: Locale) -> String {
        Bundle.module.localizedString(forKey: key, value: nil, table: nil, localizations: [locale.language])
    }

    static func number(_ value: Double?, locale: Locale) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(0...2)).locale(locale)) }
            ?? text("Unavailable", locale: locale)
    }

    static func percent(_ value: Double?, locale: Locale) -> String {
        value.map { $0.formatted(.percent.precision(.fractionLength(1)).locale(locale)) }
            ?? text("Unavailable", locale: locale)
    }

    static func day(_ date: Date, kind: ForecastKind) -> String {
        guard let calendar = try? kind.calendar() else { return "" }
        let parts = calendar.dateComponents([.year, .month, .day, .hour], from: date)
        guard let year = parts.year, let month = parts.month, let day = parts.day else { return "" }
        let date = String(format: "%04d-%02d-%02d", year, month, day)
        return kind == .seoulBike ? date + String(format: " %02d:00", parts.hour ?? 0) : date
    }

    static func failure(_ failure: ForecastFailure, kind: ForecastKind) -> String {
        if kind == .seoulBike {
            switch failure {
            case .invalidData, .sourceMismatch, .cache, .network:
                return "The bundled bike data could not be verified. Close and reopen the demo to retry."
            case .invalidRequest: return "Choose a valid whole hour and horizon."
            default: break
            }
        }
        return switch failure {
        case .invalidData, .sourceMismatch: "Sales data did not match the verified source. Load again to retry."
        case .invalidRequest: "Choose a valid product, date and horizon."
        case .network: "Could not load sales data. Check the connection and try again."
        case .responseTooLarge: "The source response exceeded the size limit."
        case .cache: "Could not read or save the sales cache. Load again to retry."
        case .unavailable: "Prepare and open the model from Models to enable prediction."
        case .predictionFailed: "Prediction failed. Try again or close and reopen the demo."
        case .busy: "Wait for the current operation to finish."
        }
    }
}
