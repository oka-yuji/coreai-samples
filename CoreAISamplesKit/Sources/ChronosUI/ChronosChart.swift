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
    }

    private var history: [Point] {
        guard let catalog = state.catalog else { return [] }
        let originStep = state.forecast?.originStep ?? state.clockStep
        guard let values = catalog.history(scenarioID: state.scenarioID, areaID: state.areaID, step: originStep)
        else { return [] }
        var points = values.suffix(8).enumerated().map { index, value in
            Point(id: index, time: catalog.time(at: originStep - 8 + index, scenarioID: state.scenarioID), value: value)
        }
        if let forecast = state.forecast {
            for lead in 0..<4 {
                let target = forecast.originStep + lead
                if let value = catalog.revealedValue(
                    scenarioID: state.scenarioID, areaID: state.areaID,
                    targetStep: target, clockStep: state.clockStep)
                {
                    points.append(
                        Point(id: 8 + lead, time: catalog.time(at: target, scenarioID: state.scenarioID), value: value))
                }
            }
        }
        return points
    }

    private func copy(_ key: String) -> String {
        String(localized: String.LocalizationValue(key), bundle: .module, locale: locale)
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
                        series: .value(copy("Series"), copy("Actual"))
                    )
                    .foregroundStyle(Color.blue)
                    .lineStyle(StrokeStyle(lineWidth: 2))
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
                        }
                        LineMark(
                            x: .value(copy("Time"), date), y: .value(copy("Forecast"), max(0, values.median[index])),
                            series: .value(copy("Series"), copy("Forecast"))
                        )
                        .foregroundStyle(Color.orange)
                        .lineStyle(StrokeStyle(lineWidth: 2.5, dash: [5, 4]))
                        PointMark(
                            x: .value(copy("Time"), date), y: .value(copy("Forecast"), max(0, values.median[index]))
                        )
                        .symbolSize(index == state.lead ? 55 : 25)
                        .foregroundStyle(Color.orange)
                    }
                    RuleMark(
                        x: .value(copy("Origin"), catalog.time(at: forecast.originStep, scenarioID: state.scenarioID))
                    )
                    .foregroundStyle(.secondary.opacity(0.4))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
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
        }
        .padding(16)
        .background(ChronosStyle.surface, in: RoundedRectangle(cornerRadius: 20))
    }
}
