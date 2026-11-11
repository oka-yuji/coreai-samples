import ChronosFeature
import SwiftUI

enum ChronosStyle {
    static let taxi = Color(red: 0.98, green: 0.72, blue: 0.14)

    static var background: Color {
        #if os(macOS)
            Color(nsColor: .windowBackgroundColor)
        #else
            Color(uiColor: .systemGroupedBackground)
        #endif
    }

    static var surface: Color {
        #if os(macOS)
            Color(nsColor: .controlBackgroundColor)
        #else
            Color(uiColor: .secondarySystemGroupedBackground)
        #endif
    }

    static func demand(_ value: Float?) -> Color {
        guard let value, value.isFinite else { return .gray.opacity(0.15) }
        return switch value {
        case ..<50: Color.blue.opacity(0.18)
        case ..<100: Color.teal.opacity(0.35)
        case ..<200: Color.yellow.opacity(0.55)
        case ..<500: Color.orange.opacity(0.65)
        default: Color.orange.opacity(0.85)
        }
    }

    static func count(_ value: Float?) -> String {
        guard let value, value.isFinite else { return "—" }
        return String(format: "%.0f", max(0, value))
    }

    static func time(_ date: Date, timeZone: String = "America/New_York") -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: timeZone)
        formatter.dateFormat = timeZone == "America/New_York" && NYCTime.isAmbiguous(date) ? "HH:mm zzz" : "HH:mm"
        return formatter.string(from: date)
    }

    static func day(_ date: Date, timeZone: String) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: timeZone)
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }

    static func interval(_ start: Date, _ end: Date, timeZone: String) -> String {
        time(start, timeZone: timeZone) + "–" + time(end, timeZone: timeZone)
    }

}
