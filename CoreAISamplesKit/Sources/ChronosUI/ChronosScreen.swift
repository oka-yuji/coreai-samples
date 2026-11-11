import ChronosFeature
import FeatureStore
import SwiftUI

public struct ChronosScreen: View {
    private let store: Store<ChronosFeature>
    @Environment(\.locale) private var locale

    public init(store: Store<ChronosFeature>) { self.store = store }
    private var state: ChronosFeature.State { store.state }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("NYC taxi pickups", bundle: .module).font(.largeTitle.bold())
                Text("Fixed January 2025 records · New York time", bundle: .module).foregroundStyle(.secondary)
                if let catalog = state.catalog {
                    controls(catalog)
                    HStack {
                        Text(
                            LocalizedStringKey(state.mode == .live ? "On-device prediction" : "Precomputed preview"),
                            bundle: .module
                        )
                        .font(.headline)
                        Spacer()
                        Text(
                            ChronosStyle.day(
                                catalog.time(at: state.clockStep, scenarioID: state.scenarioID),
                                timeZone: catalog.timeZone))
                    }
                    Text("Simulation clock · New York", bundle: .module).font(.caption)
                    Text(ChronosStyle.time(catalog.time(at: state.clockStep, scenarioID: state.scenarioID)))
                        .font(.title2.monospacedDigit())
                    if !state.liveAvailable {
                        Text(
                            "This preview uses fixed precomputed results. Open the prepared model from Models for on-device prediction.",
                            bundle: .module
                        )
                        .font(.callout).foregroundStyle(.secondary)
                    }
                    actions
                    if state.phase == .predicting || state.phase == .stopping { ProgressView() }
                    if let error = state.error {
                        Text(LocalizedStringKey(ChronosCopy.failure(error)), bundle: .module).foregroundStyle(.red)
                    }
                    interval(catalog)
                    ChronosMap(state: state, select: { store.send(.selectArea($0)) })
                    numbers
                    ChronosChart(state: state, compact: false)
                    Text(
                        "Model input: 21 days of pickup counts (1,008 half-hour bins). The chart shows the last 4 hours. One series at a time; no weather, event, zone ID or geographic inputs.",
                        bundle: .module
                    )
                    .font(.caption).foregroundStyle(.secondary)
                    Text(
                        "Counts are recorded pickups, not passengers or unmet demand. The 10–90% quantile range is not a calibrated coverage guarantee. This is not real-time data.",
                        bundle: .module
                    )
                    .font(.caption).foregroundStyle(.secondary)
                    Text(
                        "Apple basemap tiles need a network connection. The fixed records and on-device prediction work offline.",
                        bundle: .module
                    )
                    .font(.caption).foregroundStyle(.secondary)
                    if let source = URL(string: "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page") {
                        Link(destination: source) { Text("Source: NYC Taxi & Limousine Commission", bundle: .module) }
                            .font(.caption)
                    }
                } else if let error = state.error {
                    Text(LocalizedStringKey(ChronosCopy.failure(error)), bundle: .module)
                    Button {
                        store.send(.appear)
                    } label: {
                        Text("Try again", bundle: .module)
                    }
                } else {
                    ProgressView()
                }
            }
            .padding(20).frame(maxWidth: 1000, alignment: .leading).frame(maxWidth: .infinity)
        }
    }

    private func controls(_ catalog: ChronosCatalog) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker(selection: Binding(get: { state.scenarioID }, set: { store.send(.selectScenario($0)) })) {
                ForEach(catalog.scenarios) { scene in Text(ChronosCopy.name(scene.name, locale: locale)).tag(scene.id) }
            } label: {
                Text("Scene", bundle: .module)
            }
            .disabled(state.phase != .ready)
            Picker(selection: Binding(get: { state.areaID }, set: { store.send(.selectArea($0)) })) {
                ForEach(catalog.areas) { area in Text(ChronosCopy.name(area.name, locale: locale)).tag(area.id) }
            } label: {
                Text("Taxi zone", bundle: .module)
            }
            Picker(selection: Binding(get: { state.mode }, set: { store.send(.selectMode($0)) })) {
                Text("Precomputed preview", bundle: .module).tag(ChronosForecastMode.precomputed)
                if state.liveAvailable { Text("On-device prediction", bundle: .module).tag(ChronosForecastMode.live) }
            } label: {
                Text("Prediction mode", bundle: .module)
            }
            .disabled(state.phase != .ready)
        }.pickerStyle(.menu)
    }

    private var actions: some View {
        ViewThatFits(in: .horizontal) {
            HStack { predictionButtons }
            VStack(alignment: .leading) { predictionButtons }
        }
    }

    @ViewBuilder private var predictionButtons: some View {
        Button {
            store.send(.predict)
        } label: {
            Text(
                LocalizedStringKey(state.mode == .live ? "Predict on this device" : "Show precomputed forecast"),
                bundle: .module)
        }.buttonStyle(.borderedProminent).disabled(!state.canPredict)
        Button {
            store.send(.advance)
        } label: {
            Text("Advance 30 minutes", bundle: .module)
        }.disabled(!state.canAdvance)
        Button {
            store.send(.reset)
        } label: {
            Text("Reset", bundle: .module)
        }.disabled(state.phase != .ready)
        if state.phase == .predicting {
            Button {
                store.send(.cancelPrediction)
            } label: {
                Text("Cancel", bundle: .module)
            }
        }
    }

    private func interval(_ catalog: ChronosCatalog) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if state.forecast != nil {
                Picker(selection: Binding(get: { state.lead }, set: { store.send(.selectLead($0)) })) {
                    ForEach(0..<4) { lead in
                        let origin = state.forecast?.originStep ?? state.clockStep
                        Text(
                            ChronosStyle.interval(
                                catalog.time(at: origin + lead, scenarioID: state.scenarioID),
                                catalog.time(at: origin + lead + 1, scenarioID: state.scenarioID),
                                timeZone: catalog.timeZone)
                        ).tag(lead)
                    }
                } label: {
                    Text("Forecast interval", bundle: .module)
                }
                .pickerStyle(.menu)
            }
            if let start = state.displayStart, let end = state.displayEnd {
                Text(
                    ChronosStyle.day(start, timeZone: catalog.timeZone) + " · "
                        + ChronosStyle.interval(start, end, timeZone: catalog.timeZone) + " · America/New_York"
                )
                .font(.headline)
            }
        }
    }

    private var numbers: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let area = state.selectedArea { Text(ChronosCopy.name(area.name, locale: locale)).font(.title2) }
            HStack(alignment: .top, spacing: 30) {
                VStack(alignment: .leading) {
                    Text(
                        LocalizedStringKey(
                            state.forecast == nil ? "Observed trips / 30 min" : "Median forecast / 30 min"),
                        bundle: .module)
                    Text(ChronosStyle.count(state.displayedValue(areaID: state.areaID))).font(
                        .largeTitle.monospacedDigit())
                }
                if state.forecast != nil {
                    VStack(alignment: .leading) {
                        Text("Actual pickups", bundle: .module)
                        Text(ChronosStyle.count(state.selectedTruth)).font(.largeTitle.monospacedDigit())
                        if state.selectedTruth == nil { Text("Not yet elapsed", bundle: .module).font(.caption) }
                    }
                }
            }
            if let batch = state.forecast, let catalog = state.catalog {
                Text(
                    "Forecast origin: \(ChronosStyle.time(catalog.time(at: batch.originStep, scenarioID: state.scenarioID)))",
                    bundle: .module
                ).font(.caption)
            }
            if let forecast = state.selectedForecast {
                Text(
                    "10–90%: \(ChronosStyle.count(forecast.lower[state.lead]))–\(ChronosStyle.count(forecast.upper[state.lead]))",
                    bundle: .module)
            }
            if let seconds = state.forecast?.processingSeconds {
                Text(
                    "Device processing, including preparation when needed: \(seconds.formatted(.number.precision(.fractionLength(2)))) s", bundle: .module
                )
                .font(.caption)
            }
        }
    }
}

enum ChronosCopy {
    static func name(_ name: ChronosDisplayName, locale: Locale) -> String {
        locale.language.languageCode?.identifier == "ja" ? name.ja : name.en
    }

    static func failure(_ error: ChronosDemoFailure) -> String {
        switch error {
        case .invalidData: "The fixed data or precomputed results failed validation. Reinstall the app resources."
        case .unavailable: "The managed model is unavailable. Reopen the demo from Models."
        case .busy: "The previous operation is still finishing. Try again."
        case .predictionFailed:
            "Prediction failed. The prediction mode has not changed. Try again or select precomputed preview."
        }
    }
}
