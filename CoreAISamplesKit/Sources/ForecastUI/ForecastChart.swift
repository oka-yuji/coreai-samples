import Charts
import ForecastFeature
import SwiftUI

struct ForecastChart: View {
    struct Day: Identifiable {
        let id: Int
        let label: String
        let history: Float?
        let actual: Float?
        let baseline: Float?
        let median: Float?
        let lower: Float?
        let upper: Float?
        let baselineSegment: Int
    }

    let state: ForecastFeature.State
    @Environment(\.locale) private var locale

    private var days: [Day] {
        guard let dataset = state.dataset,
            let series = dataset.series.first(where: { $0.id == state.seriesID }),
            let request = state.request
        else { return [] }
        let start = max(0, state.originIndex - 56)
        var segment = 0
        return (start..<(state.originIndex + state.horizon)).map { index in
            let step = index - state.originIndex
            let future = step >= 0
            let result = state.result
            let baseline = future && result != nil ? request.baseline[step] : nil
            if baseline == nil { segment += 1 }
            return Day(
                id: index, label: ForecastCopy.day(dataset.timestamps[index], kind: dataset.kind),
                history: future ? nil : series.values[index],
                actual: future && result != nil ? request.actuals[step] : nil,
                baseline: baseline, median: future ? result?.median[step] : nil,
                lower: future ? result?.lower[step] : nil, upper: future ? result?.upper[step] : nil,
                baselineSegment: segment)
        }
    }

    private var visibleDomain: ClosedRange<Int> {
        guard let first = days.first, let last = days.last else { return 0...1 }
        return first.id...max(first.id + 1, last.id)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Sales by calendar date", bundle: .module).font(.headline)
            legend
            Chart(days) { day in
                if let history = day.history {
                    PointMark(x: .value("Date", day.id), y: .value("Sales", history))
                        .foregroundStyle(.secondary).symbolSize(16)
                }
                if let lower = day.lower, let upper = day.upper {
                    AreaMark(x: .value("Date", day.id), yStart: .value("P10", lower), yEnd: .value("P90", upper))
                        .foregroundStyle(.blue.opacity(0.15))
                }
                if let median = day.median {
                    LineMark(x: .value("Date", day.id), y: .value("Median", median), series: .value("Series", "median"))
                        .foregroundStyle(.blue).lineStyle(StrokeStyle(lineWidth: 2))
                }
                if let baseline = day.baseline {
                    LineMark(
                        x: .value("Date", day.id), y: .value("Baseline", baseline),
                        series: .value("Series", "baseline-\(day.baselineSegment)")
                    )
                    .foregroundStyle(.orange).lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 4]))
                    PointMark(x: .value("Date", day.id), y: .value("Baseline", baseline))
                        .foregroundStyle(.orange).symbol(.diamond).symbolSize(16)
                }
                if let actual = day.actual {
                    PointMark(x: .value("Date", day.id), y: .value("Actual", actual))
                        .foregroundStyle(.primary).symbol(.square).symbolSize(30)
                }
            }
            .chartXScale(domain: visibleDomain)
            .chartXAxis {
                AxisMarks(values: days.filter { $0.id % 14 == state.originIndex % 14 }.map(\.id)) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel {
                        if let index = value.as(Int.self), let day = days.first(where: { $0.id == index }) {
                            Text(day.label).font(.caption2)
                        }
                    }
                }
            }
            .chartYAxis { AxisMarks(position: .trailing) }
            .frame(height: 280)
            .accessibilityRepresentation {
                VStack(alignment: .leading) {
                    Text("Sales by calendar date · Europe/Rome", bundle: .module)
                    ForEach(days) { day in
                        Text(verbatim: accessibility(day)).accessibilityElement(children: .ignore)
                            .accessibilityLabel(accessibility(day))
                    }
                }.accessibilityElement(children: .contain)
            }
            Text("History shows the last 8 weeks. Gaps are missing observations.", bundle: .module)
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: 5) {
            Label {
                Text("History · circles", bundle: .module)
            } icon: {
                Image(systemName: "circle.fill").foregroundStyle(.secondary)
            }
            Label {
                Text("Actuals · squares", bundle: .module)
            } icon: {
                Image(systemName: "square.fill")
            }
            Label {
                Text("Median and P10–P90 band", bundle: .module)
            } icon: {
                Image(systemName: "line.diagonal").foregroundStyle(.blue)
            }
            Label {
                Text("7-day seasonal naive · dashed", bundle: .module)
            } icon: {
                Image(systemName: "diamond.fill").foregroundStyle(.orange)
            }
        }.font(.caption)
    }

    private func accessibility(_ day: Day) -> String {
        let values: [(String, Float?)] =
            day.id < state.originIndex
            ? [("History", day.history)]
            : [
                ("Actuals", day.actual), ("Median", day.median), ("P10", day.lower), ("P90", day.upper),
                ("Seasonal naive", day.baseline),
            ]
        return "\(day.label) Europe/Rome, \(state.seriesID). "
            + values.map { name, value in
                ForecastCopy.text(name, locale: locale) + ": "
                    + (value.map { ForecastCopy.number(Double($0), locale: locale) }
                        ?? ForecastCopy.text("Missing", locale: locale))
            }.joined(separator: "; ")
    }
}
