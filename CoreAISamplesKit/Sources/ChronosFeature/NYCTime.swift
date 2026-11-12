import Foundation

public enum NYCTime {
    public static func isSupported(_ date: Date) -> Bool {
        let seconds = date.timeIntervalSince1970
        return seconds.isFinite && (-62_135_596_800..<253_402_300_800).contains(seconds)
    }

    public static func isValidWindow(start: Date, count: Int) -> Bool {
        guard (1...1024).contains(count), isSupported(start), start == aligned(start) else { return false }
        let duration = Double(count) * 1800
        let end = start.addingTimeInterval(duration)
        return isSupported(end) && end > start && end.timeIntervalSince(start) == duration
    }

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
        guard isSupported(day), isSupported(time) else { return nil }
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
        guard isSupported(date) else { return [] }
        let start = calendar.startOfDay(for: date)
        guard let end = calendar.date(byAdding: .day, value: 1, to: start), isSupported(start), isSupported(end),
            let count = Int(exactly: end.timeIntervalSince(start) / 1800), (1...50).contains(count)
        else { return [] }
        return (0..<count).map { start.addingTimeInterval(Double($0 * 1800)) }
    }
}
