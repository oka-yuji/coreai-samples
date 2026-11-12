import Charts
import ChronosFeature
import SwiftUI

struct ChronosChart: View {
    @Environment(\.locale) private var locale
    let state: ChronosFeature.State
    let compact: Bool

    private struct Point: Identifiable {
        let id: Int
        let time: Date
        let value: Float
        let segment: Int
    }

    private var history: [Point] {
        guard let catalog = state.catalog else { return [] }
        let originStep = state.forecast?.originStep ?? state.clockStep
        guard let values = catalog.history(scenarioID: state.scenarioID, areaID: state.areaID, step: originStep)
        else { return [] }
        var segment = 0
        var points: [Point] = []
        for (index, value) in values.suffix(8).enumerated() {
            guard value.isFinite else {
                segment += 1
                continue
            }
            points.append(
                Point(
                    id: index, time: catalog.time(at: originStep - 8 + index, scenarioID: state.scenarioID),
                    value: value, segment: segment))
        }
        if let forecast = state.forecast {
            for lead in 0..<4 {
                let target = forecast.originStep + lead
                let value: Float?
                if let actuals = state.selectedScenario?.apiActuals?.series.first(where: { $0.areaID == state.areaID }),
                    actuals.values.indices.contains(target)
                {
                    value = actuals.values[target]
                } else {
                    value = catalog.revealedValue(
                        scenarioID: state.scenarioID, areaID: state.areaID, targetStep: target,
                        clockStep: state.clockStep)
                }
                if let value {
                    points.append(
                        Point(
                            id: 8 + lead, time: catalog.time(at: target, scenarioID: state.scenarioID), value: value,
                            segment: segment))
                } else {
                    segment += 1
                }
            }
        }
        return points
    }

    private var chartTimeZone: TimeZone {
        TimeZone(identifier: state.catalog?.timeZone ?? "America/New_York") ?? .gmt
    }

    private var chartCalendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = chartTimeZone
        return calendar
    }

    private func intervalLabel(_ date: Date, kind: String) -> Text {
        let zone = state.catalog?.timeZone ?? "America/New_York"
        let interval =
            ChronosStyle.day(date, timeZone: zone) + " "
            + ChronosStyle.interval(date, date.addingTimeInterval(1800), timeZone: zone)
            + " · America/New_York · "
        return Text(verbatim: interval) + Text(LocalizedStringKey(kind), bundle: .module)
    }

    private func copy(_ key: String) -> String {
        String(localized: String.LocalizationValue(key), bundle: .module, locale: locale)
    }

    private var accessibleValues: some View {
        VStack {
            ForEach(history) { point in
                Rectangle()
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(intervalLabel(point.time, kind: "Actual pickups"))
                    .accessibilityValue(Text(verbatim: ChronosStyle.count(point.value)))
            }
            if let forecast = state.forecast, let values = state.selectedForecast, let catalog = state.catalog {
                ForEach(0..<4, id: \.self) { index in
                    let date = catalog.time(at: forecast.originStep + index, scenarioID: state.scenarioID)
                    Rectangle()
                        .accessibilityElement(children: .ignore)
                        .accessibilityLabel(intervalLabel(date, kind: "Forecast"))
                        .accessibilityValue(
                            Text(
                                "Median \(ChronosStyle.count(values.median[index])), 10–90%: \(ChronosStyle.count(values.lower[index]))–\(ChronosStyle.count(values.upper[index]))",
                                bundle: .module))
                }
            }
        }
        .accessibilityElement(children: .contain)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Last 4 hours and next 2 hours", bundle: .module).font(.headline)
                Spacer()
                Text("Trips / 30 min", bundle: .module).font(.caption).foregroundStyle(.secondary)
            }
            Chart {
                ForEach(history) { point in
                    LineMark(
                        x: .value(copy("Time"), point.time), y: .value(copy("Trips"), point.value),
                        series: .value(copy("Series"), String(point.segment))
                    )
                    .foregroundStyle(Color.blue)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .accessibilityLabel(intervalLabel(point.time, kind: "Actual pickups"))
                    .accessibilityValue(Text(verbatim: ChronosStyle.count(point.value)))
                }
                if let forecast = state.forecast, let values = state.selectedForecast, let catalog = state.catalog {
                    ForEach(0..<4, id: \.self) { index in
                        let date = catalog.time(at: forecast.originStep + index, scenarioID: state.scenarioID)
                        if values.lower[index] <= values.upper[index] {
                            AreaMark(
                                x: .value(copy("Time"), date),
                                yStart: .value(copy("Lower"), max(0, values.lower[index])),
                                yEnd: .value(copy("Upper"), max(0, values.upper[index]))
                            )
                            .foregroundStyle(ChronosStyle.taxi.opacity(0.2))
                            .accessibilityHidden(true)
                        }
                        LineMark(
                            x: .value(copy("Time"), date), y: .value(copy("Forecast"), max(0, values.median[index])),
                            series: .value(copy("Series"), copy("Forecast"))
                        )
                        .foregroundStyle(Color.orange)
                        .lineStyle(StrokeStyle(lineWidth: 2.5, dash: [5, 4]))
                        .accessibilityLabel(intervalLabel(date, kind: "Forecast"))
                        .accessibilityValue(
                            Text(
                                "Median \(ChronosStyle.count(values.median[index])), 10–90%: \(ChronosStyle.count(values.lower[index]))–\(ChronosStyle.count(values.upper[index]))",
                                bundle: .module))
                        PointMark(
                            x: .value(copy("Time"), date), y: .value(copy("Forecast"), max(0, values.median[index]))
                        )
                        .symbolSize(index == state.lead ? 55 : 25)
                        .foregroundStyle(Color.orange)
                        .accessibilityHidden(true)
                    }
                    RuleMark(
                        x: .value(copy("Origin"), catalog.time(at: forecast.originStep, scenarioID: state.scenarioID))
                    )
                    .foregroundStyle(.secondary.opacity(0.4))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                    .accessibilityHidden(true)
                }
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: compact ? 4 : 6)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let date = value.as(Date.self) {
                            Text(ChronosStyle.time(date, timeZone: state.catalog?.timeZone ?? "America/New_York"))
                        }
                    }
                }
            }
            .chartYAxis { AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) }
            .chartYScale(domain: .automatic(includesZero: true))
            .frame(height: compact ? 190 : 250)
            .accessibilityLabel(Text("History and forecast chart", bundle: .module))
            .accessibilityChildren { accessibleValues }
            .environment(\.timeZone, chartTimeZone)
            .environment(\.calendar, chartCalendar)
            HStack(spacing: 16) {
                Label {
                    Text("Actual pickups", bundle: .module)
                } icon: {
                    Image(systemName: "minus")
                }.foregroundStyle(Color.blue)
                Label {
                    Text("Forecast", bundle: .module)
                } icon: {
                    Image(systemName: "line.diagonal")
                }.foregroundStyle(.orange)
                    .symbolRenderingMode(.hierarchical)
                Spacer(minLength: 0)
            }
            .font(.caption)
            Text("Shaded area: model 10–90% quantiles", bundle: .module).font(.caption2).foregroundStyle(.secondary)
            if state.selectedScenario?.apiActuals?.availability.contains(.unverified) == true {
                Text("Unverified actuals are missing from the chart and cannot advance the forecast.", bundle: .module)
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .background(ChronosStyle.surface, in: RoundedRectangle(cornerRadius: 20))
    }
}
