import Foundation

extension ForecastDataset {
    public func origins(on day: Date, horizon: Int) -> ClosedRange<Int>? {
        guard day.timeIntervalSince1970.isFinite, abs(day.timeIntervalSince1970) < 100_000_000_000,
            let range = origins(horizon: horizon), let calendar = try? kind.calendar()
        else { return nil }
        let start = calendar.startOfDay(for: day)
        guard let end = calendar.date(byAdding: .day, value: 1, to: start),
            let first = range.first(where: { timestamps[$0] >= start && timestamps[$0] < end }),
            let last = range.last(where: { timestamps[$0] >= start && timestamps[$0] < end })
        else { return nil }
        return first...last
    }
}
