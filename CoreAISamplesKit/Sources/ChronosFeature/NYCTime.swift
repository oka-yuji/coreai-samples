import Foundation

public enum NYCTime {
    public static var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/New_York") ?? .gmt
        return calendar
    }

    public static func yearStart(_ year: Int) -> Date? {
        calendar.date(from: DateComponents(year: year, month: 1, day: 1))
    }

    public static func year(_ date: Date) -> Int { calendar.component(.year, from: date) }

    public static func aligned(_ date: Date) -> Date {
        Date(timeIntervalSince1970: floor(date.timeIntervalSince1970 / 1800) * 1800)
    }

    public static func key(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return formatter.string(from: date)
    }

    public static func isAmbiguous(_ date: Date) -> Bool {
        let value = key(date)
        return value == key(date.addingTimeInterval(-3600)) || value == key(date.addingTimeInterval(3600))
    }

    public static func matchingSlot(on day: Date, preserving time: Date, in range: ClosedRange<Date>? = nil) -> Date? {
        let parts = calendar.dateComponents([.hour, .minute], from: time)
        let options = slots(on: day).filter {
            calendar.component(.hour, from: $0) == parts.hour && calendar.component(.minute, from: $0) == parts.minute
                && (range?.contains($0) ?? true)
        }
        return options.first {
            calendar.timeZone.secondsFromGMT(for: $0) == calendar.timeZone.secondsFromGMT(for: time)
        }
            ?? options.first
    }

    public static func slots(on date: Date) -> [Date] {
        let start = calendar.startOfDay(for: date)
        guard let end = calendar.date(byAdding: .day, value: 1, to: start) else { return [] }
        return (0..<Int(end.timeIntervalSince(start) / 1800)).map { start.addingTimeInterval(Double($0 * 1800)) }
    }
}
