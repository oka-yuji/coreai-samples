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
            Text("Product sales forecast", bundle: .module).font(.title.bold())
            Text("Daily observed sales · Europe/Rome", bundle: .module).foregroundStyle(.secondary)
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
                Text(LocalizedStringKey(ForecastCopy.failure(error)), bundle: .module)
                    .foregroundStyle(.red).accessibilityIdentifier("forecast-error")
            }
            Text(
                "Observed sales are not unmet demand. Missing dates are not zero or assumed store closures. The P10–P90 range is not a coverage guarantee.",
                bundle: .module
            ).font(.caption).foregroundStyle(.secondary)
            Text(
                "Input: the previous 365 calendar days, without future sales or promotions. This model supports 7 or 14 displayed days; 28 days and covariates are not supported.",
                bundle: .module
            ).font(.caption).foregroundStyle(.secondary)
        }
        .padding(20).frame(maxWidth: 1000, alignment: .leading).frame(maxWidth: .infinity)
    }

    private var source: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                store.send(.loadData)
            } label: {
                Text("Load public sales data", bundle: .module)
            }
            .disabled(!state.active || state.isWorking || state.phase == .stopping)
            if let url = URL(string: "https://doi.org/10.17632/njdkntcpc9.1") {
                Link(destination: url) {
                    Text("Source: Hierarchical Sales Data · Mendeley Data v1", bundle: .module)
                }
            }
            Text(
                "Noncommercial data · CC BY-NC 3.0 at Mendeley; UCI lists CC BY 4.0. Loading stores the verified original in this app’s cache for offline use.",
                bundle: .module
            ).font(.caption).foregroundStyle(.secondary)
        }
    }

    private func controls(_ dataset: ForecastDataset) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker(selection: Binding(get: { state.seriesID }, set: { store.send(.selectSeries($0)) })) {
                ForEach(dataset.series, id: \.id) { Text(verbatim: $0.id).tag($0.id) }
            } label: {
                Text("Product", bundle: .module)
            }
            if let range = dataset.origins(horizon: state.horizon),
                let calendar = try? dataset.kind.calendar()
            {
                DatePicker(
                    selection: Binding(
                        get: { dataset.timestamps[state.originIndex] },
                        set: { date in
                            if let index = dataset.timestamps.firstIndex(of: calendar.startOfDay(for: date)) {
                                store.send(.selectOrigin(index))
                            }
                        }),
                    in: dataset.timestamps[range.lowerBound]...dataset.timestamps[range.upperBound],
                    displayedComponents: .date
                ) { Text("Forecast start · Europe/Rome", bundle: .module) }
                .environment(\.calendar, calendar).environment(\.timeZone, calendar.timeZone)
            }
            Picker(selection: Binding(get: { state.horizon }, set: { store.send(.selectHorizon($0)) })) {
                Text("7 days", bundle: .module).tag(7)
                Text("14 days", bundle: .module).tag(14)
            } label: {
                Text("Horizon", bundle: .module)
            }
            .pickerStyle(.segmented)
        }.disabled(!state.canSelect)
    }

    private func metrics(_ result: ForecastResult) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Compared dates: \(result.metrics.count)", bundle: .module).font(.headline)
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
                "Runtime includes validation and model loading when needed. Metrics use the same dates with actuals, median and baseline. WAPE is unavailable when actuals sum to zero. Negative predictions are shown as zero; raw output is preserved.",
                bundle: .module
            ).font(.caption).foregroundStyle(.secondary)
        }
    }
}
