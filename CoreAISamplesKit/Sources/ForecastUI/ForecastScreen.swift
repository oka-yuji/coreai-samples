import FeatureStore
import ForecastFeature
import ForecastRuntime
import SwiftUI

public struct ForecastScreen: View {
    private let store: Store<ForecastFeature>
    private let scrollsContent: Bool
    @Environment(\.locale) private var locale

    public init(store: Store<ForecastFeature>, scrollsContent: Bool = true) {
        self.store = store
        self.scrollsContent = scrollsContent
    }

    private var state: ForecastFeature.State { store.state }

    public var body: some View {
        #if os(macOS)
            if scrollsContent { ScrollView { contents } } else { contents }
        #else
            contents
        #endif
    }

    private var contents: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text(
                LocalizedStringKey(state.kind == .productSales ? "Product sales forecast" : "Seoul Bike forecast"),
                bundle: .module
            ).font(.title.bold())
            Text(
                LocalizedStringKey(
                    state.kind == .productSales
                        ? "Daily observed sales · Europe/Rome" : "Hourly observed rentals · Asia/Seoul"),
                bundle: .module
            ).foregroundStyle(.secondary)
            source
            if let dataset = state.dataset {
                controls(dataset)
                HStack {
                    Button {
                        store.send(.predict)
                    } label: {
                        Text("Predict on this device", bundle: .module)
                    }
                    .buttonStyle(.borderedProminent).disabled(!state.canPredict)
                    if state.isWorking {
                        Button {
                            store.send(.stop)
                        } label: {
                            Text("Stop", bundle: .module)
                        }
                    }
                }
                if !state.liveAvailable {
                    Text("Prepare and open the model from Models to enable prediction.", bundle: .module)
                        .font(.callout).foregroundStyle(.secondary)
                }
                ForecastChart(state: state)
                if let result = state.result { metrics(result) }
                if state.kind == .seoulBike,
                    let weather = try? dataset.pastWeather(originIndex: state.originIndex, count: 72)
                {
                    ForecastWeatherChart(observations: weather)
                }
            }
            if state.isWorking || state.phase == .stopping {
                ProgressView {
                    Text(
                        LocalizedStringKey(state.phase == .stopping ? "Finishing the current operation…" : "Working…"),
                        bundle: .module)
                }
                if state.dataset == nil, state.isWorking {
                    Button {
                        store.send(.stop)
                    } label: {
                        Text("Stop", bundle: .module)
                    }
                }
            }
            if let error = state.error {
                Text(LocalizedStringKey(ForecastCopy.failure(error, kind: state.kind)), bundle: .module)
                    .foregroundStyle(.red).accessibilityIdentifier("forecast-error")
            }
            limitations
        }
        .padding(20).frame(maxWidth: 1000, alignment: .leading).frame(maxWidth: .infinity)
    }

    private var source: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                store.send(.loadData)
            } label: {
                Text(
                    LocalizedStringKey(
                        state.kind == .productSales ? "Load public sales data" : "Load bundled bike data"),
                    bundle: .module)
            }
            .disabled(!state.active || state.isWorking || state.phase == .stopping)
            if state.kind == .seoulBike {
                BikeAttribution()
            } else {
                salesAttribution
            }
        }
    }

    private var salesAttribution: some View {
        Group {
            if let url = URL(string: "https://doi.org/10.17632/njdkntcpc9.1") {
                Link(destination: url) {
                    Text("Source: Hierarchical Sales Data · Mendeley Data v1", bundle: .module)
                }
            }
            Text("Paolo Mancuso, Veronica Piccialli and Antonio M. Sudoso (2021)", bundle: .module).font(.caption)
                .foregroundStyle(.secondary)
            Text(
                "Noncommercial data · CC BY-NC 3.0 at Mendeley; UCI lists CC BY 4.0. Loading stores the verified original in this app’s cache for offline use.",
                bundle: .module
            ).font(.caption).foregroundStyle(.secondary)
        }
    }

    private func controls(_ dataset: ForecastDataset) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            if state.kind == .productSales {
                Picker(selection: Binding(get: { state.seriesID }, set: { store.send(.selectSeries($0)) })) {
                    ForEach(dataset.series, id: \.id) { Text(verbatim: $0.id).tag($0.id) }
                } label: {
                    Text("Product", bundle: .module)
                }
            }
            if let range = dataset.origins(horizon: state.horizon),
                let calendar = try? dataset.kind.calendar()
            {
                let firstDay = calendar.startOfDay(for: dataset.timestamps[range.lowerBound])
                let lastDay = calendar.startOfDay(for: dataset.timestamps[range.upperBound])
                DatePicker(
                    selection: Binding(
                        get: { calendar.startOfDay(for: dataset.timestamps[state.originIndex]) },
                        set: { date in
                            if let hours = dataset.origins(on: date, horizon: state.horizon) {
                                let oldHour = calendar.component(.hour, from: dataset.timestamps[state.originIndex])
                                let firstHour = calendar.component(.hour, from: dataset.timestamps[hours.lowerBound])
                                let offset = state.kind == .productSales ? 0 : max(0, oldHour - firstHour)
                                store.send(.selectOrigin(min(hours.lowerBound + offset, hours.upperBound)))
                            }
                        }),
                    in: firstDay...lastDay,
                    displayedComponents: .date
                ) {
                    Text(
                        LocalizedStringKey(
                            state.kind == .productSales ? "Forecast start · Europe/Rome" : "Forecast start · Asia/Seoul"
                        ),
                        bundle: .module)
                }
                .environment(\.calendar, calendar).environment(\.timeZone, calendar.timeZone)
                if state.kind == .seoulBike,
                    let hours = dataset.origins(on: dataset.timestamps[state.originIndex], horizon: state.horizon)
                {
                    Picker(selection: Binding(get: { state.originIndex }, set: { store.send(.selectOrigin($0)) })) {
                        ForEach(Array(hours), id: \.self) { index in
                            Text(
                                verbatim: String(
                                    format: "%02d:00", calendar.component(.hour, from: dataset.timestamps[index]))
                            )
                            .tag(index)
                        }
                    } label: {
                        Text("Hour · Asia/Seoul", bundle: .module)
                    }
                }
            }
            Picker(selection: Binding(get: { state.horizon }, set: { store.send(.selectHorizon($0)) })) {
                if state.kind == .productSales {
                    Text("7 days", bundle: .module).tag(7)
                    Text("14 days", bundle: .module).tag(14)
                } else {
                    Text("1 h", bundle: .module).tag(1)
                    Text("4 h", bundle: .module).tag(4)
                    Text("8 h", bundle: .module).tag(8)
                    Text("16 h", bundle: .module).tag(16)
                }
            } label: {
                Text("Horizon", bundle: .module)
            }
            .pickerStyle(.segmented)
        }.disabled(!state.canSelect)
    }

    private func metrics(_ result: ForecastResult) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if state.kind == .productSales {
                Text("Compared dates: \(result.metrics.count)", bundle: .module).font(.headline)
            } else {
                Text("Compared hours: \(result.metrics.count) / \(result.request.horizon)", bundle: .module).font(
                    .headline)
            }
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow {
                    Text("Forecast", bundle: .module)
                    Text("MAE", bundle: .module)
                    Text("WAPE", bundle: .module)
                }
                GridRow {
                    Text("Median", bundle: .module)
                    Text(ForecastCopy.number(result.metrics.modelMAE, locale: locale))
                    Text(ForecastCopy.percent(result.metrics.modelWAPE, locale: locale))
                }
                GridRow {
                    Text("Seasonal naive", bundle: .module)
                    Text(ForecastCopy.number(result.metrics.baselineMAE, locale: locale))
                    Text(ForecastCopy.percent(result.metrics.baselineWAPE, locale: locale))
                }
            }.font(.callout).monospacedDigit()
            Text("Runtime: \(result.seconds, format: .number.precision(.fractionLength(2))) s", bundle: .module)
                .font(.caption)
            Text(
                LocalizedStringKey(
                    state.kind == .productSales
                        ? "Runtime includes validation and model loading when needed. Metrics use the same dates with actuals, median and baseline. WAPE is unavailable when actuals sum to zero. Negative predictions use zero in both the chart and metrics; raw output is preserved."
                        : "Runtime includes validation and model loading when needed. Metrics compare the same hours with actuals, median and baseline. WAPE is unavailable when actuals sum to zero. Negative predictions use zero in the chart and metrics; all raw quantiles are preserved."
                ),
                bundle: .module
            ).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var limitations: some View {
        VStack(alignment: .leading, spacing: 8) {
            if state.kind == .productSales {
                Text(
                    "Observed sales are not unmet demand. Missing dates are not zero or assumed store closures. The P10–P90 range is not a coverage guarantee.",
                    bundle: .module)
                Text(
                    "Input: the previous 365 calendar days, without future sales or promotions. This model supports 7 or 14 displayed days; 28 days and covariates are not supported.",
                    bundle: .module)
            } else {
                Text(
                    "Shutdown zeros are observed rentals, not zero latent demand. The P10–P90 range is not a coverage guarantee.",
                    bundle: .module)
                Text(
                    "Input: only the preceding 336 hourly rental counts. Horizons are 1, 4, 8 or 16 hours. Weather inputs and 24-hour or recursive forecasts are not supported.",
                    bundle: .module)
            }
        }.font(.caption).foregroundStyle(.secondary)
    }
}
