import Charts
import ForecastFeature
import SwiftUI

struct ForecastWeatherChart: View {
    let observations: [ForecastWeatherObservation]
    @Environment(\.locale) private var locale

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Past observed weather · last 72 hours", bundle: .module).font(.headline)
            Text("Past observations only: not model input and not a future weather forecast.", bundle: .module)
                .font(.callout).foregroundStyle(.secondary)
            weatherPlot(rain: false)
            weatherPlot(rain: true)
            if let last = observations.last {
                Text(
                    "Last observation · \(ForecastCopy.day(last.timestamp, kind: .seoulBike)) Asia/Seoul",
                    bundle: .module)
                Text(
                    LocalizedStringKey(
                        last.isFunctioning ? "Operating: yes" : "Operating: no · observed rentals are zero"),
                    bundle: .module)
                Text(LocalizedStringKey(last.isHoliday ? "Holiday: yes" : "Holiday: no"), bundle: .module)
            }
        }.font(.caption)
    }

    private func weatherPlot(rain: Bool) -> some View {
        VStack(alignment: .leading) {
            Text(LocalizedStringKey(rain ? "Rainfall (mm)" : "Temperature (°C)"), bundle: .module).font(.subheadline)
            Chart(Array(observations.enumerated()), id: \.offset) { index, item in
                if rain {
                    BarMark(x: .value("Hour", index), y: .value("mm", item.rainfall)).foregroundStyle(.teal)
                } else {
                    LineMark(x: .value("Hour", index), y: .value("°C", item.temperature)).foregroundStyle(.purple)
                }
            }
            .chartXScale(domain: 0...max(1, observations.count - 1))
            .chartXAxis {
                AxisMarks(values: observations.indices.filter { $0 % 24 == 0 }) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let index = value.as(Int.self), observations.indices.contains(index) {
                            Text(verbatim: ForecastCopy.day(observations[index].timestamp, kind: .seoulBike))
                                .font(.caption2)
                        }
                    }
                }
            }
            .chartYAxis { AxisMarks(position: .trailing) }
            .frame(height: 150)
            .accessibilityRepresentation {
                VStack {
                    Text(
                        LocalizedStringKey(rain ? "Rainfall (mm) · Asia/Seoul" : "Temperature (°C) · Asia/Seoul"),
                        bundle: .module)
                    ForEach(observations, id: \.timestamp) { item in
                        Text(verbatim: label(item, rain: rain))
                            .accessibilityElement(children: .ignore).accessibilityLabel(label(item, rain: rain))
                    }
                }.accessibilityElement(children: .contain)
            }
        }
    }

    private func label(_ item: ForecastWeatherObservation, rain: Bool) -> String {
        ForecastCopy.day(item.timestamp, kind: .seoulBike) + " Asia/Seoul, "
            + ForecastCopy.text(rain ? "Rainfall (mm)" : "Temperature (°C)", locale: locale) + ": "
            + ForecastCopy.number(Double(rain ? item.rainfall : item.temperature), locale: locale)
    }
}
