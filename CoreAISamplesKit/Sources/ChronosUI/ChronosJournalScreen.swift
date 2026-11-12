import ChronosFeature
import FeatureStore
import SwiftUI

struct ChronosJournalControl: View {
    let store: Store<ChronosJournalFeature>
    @Environment(\.locale) private var locale

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                store.send(.open)
            } label: {
                Label {
                    Text("Forecast history", bundle: .module)
                } icon: {
                    Image(systemName: "clock")
                }
            }
            if store.state.saving {
                ProgressView { Text("Saving forecast…", bundle: .module) }
            } else if store.state.saveError != nil {
                Text(
                    "The forecast is ready, but could not be saved. Check storage access and available space, then retry.",
                    bundle: .module
                )
                .foregroundStyle(.red)
                Button {
                    store.send(.retrySave)
                } label: {
                    Text("Retry saving", bundle: .module)
                }
            } else if store.state.savedID != nil {
                Text("Forecast saved on this device", bundle: .module).font(.caption).foregroundStyle(.secondary)
            }
        }
        .sheet(isPresented: Binding(get: { store.state.presented }, set: { if !$0 { store.send(.close) } })) {
            ChronosJournalScreen(store: store)
                .environment(\.locale, locale)
        }
    }
}

struct ChronosJournalScreen: View {
    let store: Store<ChronosJournalFeature>
    @Environment(\.dismiss) private var dismiss
    private var state: ChronosJournalFeature.State { store.state }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if state.busy { ProgressView() }
                    if let error = state.error {
                        Text(LocalizedStringKey(failure(error)), bundle: .module).foregroundStyle(.red)
                    }
                    if let detail = state.detail {
                        record(detail)
                    } else if state.selectedID == nil {
                        listing
                    }
                }.padding(20).frame(maxWidth: 700, alignment: .leading).frame(maxWidth: .infinity)
            }
            .navigationTitle(Text("Forecast history", bundle: .module))
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    if state.selectedID != nil {
                        Button {
                            store.send(.back)
                        } label: {
                            Text("Back", bundle: .module)
                        }
                    } else {
                        Button {
                            store.send(.close)
                            dismiss()
                        } label: {
                            Text("Done", bundle: .module)
                        }
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        store.send(.open)
                    } label: {
                        Text("Reload records", bundle: .module)
                    }
                }
            }
        }
        .frame(minWidth: 300, minHeight: 400)
        .onDisappear { store.send(.close) }
    }

    private var listing: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !state.busy, state.entries.isEmpty {
                Text(
                    "No saved forecasts yet. Successful on-device predictions are saved automatically.", bundle: .module
                )
            }
            ForEach(state.entries) { entry in
                Button {
                    store.send(.select(entry.id))
                } label: {
                    VStack(alignment: .leading, spacing: 4) {
                        if let origin = entry.origin, let savedAt = entry.savedAt, let source = entry.source {
                            Text(newYork(origin))
                            Text(LocalizedStringKey(sourceName(source)), bundle: .module)
                            Text("Saved: \(savedAt.formatted())", bundle: .module).font(.caption)
                        } else {
                            Text("This record could not be read", bundle: .module).foregroundStyle(.red)
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }.buttonStyle(.bordered)
            }
        }
    }

    private func record(_ detail: ChronosJournalDetail) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(newYork(detail.record.input.snapshot.origin)).font(.headline)
            Text(LocalizedStringKey(sourceName(detail.record.input.snapshot.source)), bundle: .module)
            Text("Saved: \(detail.record.savedAt.formatted())", bundle: .module)
            Text(
                "Model version: \(String(detail.record.identity.source.revision.prefix(8))) · \(String(detail.record.identity.manifest.sha256.prefix(8)))",
                bundle: .module
            )
            .font(.caption)
            Text("Saved times use this device’s clock and are not independently certified.", bundle: .module)
                .font(.caption).foregroundStyle(.secondary)
            Button {
                store.send(.check)
            } label: {
                Text("Check published actuals", bundle: .module)
            }
            .buttonStyle(.borderedProminent).disabled(state.busy)
            Text(
                "Checks compare the first 2 hours across 20 taxi zones. Each check is kept separately.", bundle: .module
            )
            .font(.caption).foregroundStyle(.secondary)
            if detail.checks.isEmpty {
                Text("Recorded · Not checked", bundle: .module)
            }
            ForEach(detail.checks) { check in
                Divider()
                Text("Checked: \(check.checkedAt.formatted())", bundle: .module).font(.caption)
                Text(LocalizedStringKey(status(check.status)), bundle: .module).font(.headline)
                Text("Compared: \(check.metrics.comparedCount) / \(check.metrics.expectedCount)", bundle: .module)
                if let mae = check.metrics.meanAbsoluteError {
                    Text(
                        "Mean absolute error: \(mae.formatted(.number.precision(.fractionLength(2))))", bundle: .module)
                }
                if let wape = check.metrics.weightedAbsolutePercentageError {
                    Text(
                        "Weighted absolute percentage error: \(wape.formatted(.number.precision(.fractionLength(2))))%",
                        bundle: .module)
                } else {
                    Text(
                        "Percentage error is unavailable when there are no pairs or the actual total is zero.",
                        bundle: .module
                    )
                    .font(.caption)
                }
                missing(check, .unpublished, "Not published")
                missing(check, .unverified, "Not independently verified")
                missing(check, .ambiguousTime, "Missing: repeated New York time")
            }
        }
    }

    @ViewBuilder private func missing(_ check: ChronosActualCheck, _ reason: SlotAvailability, _ title: String)
        -> some View
    {
        let count = check.missingPairs(reason)
        if count > 0 {
            HStack {
                Text(LocalizedStringKey(title), bundle: .module)
                Text(count, format: .number)
            }.font(.caption)
        }
    }

    private func sourceName(_ source: ChronosInputCapture.Source) -> String {
        source == .api ? "Source: NYC public API" : "Source: bundled NYC records"
    }

    private func newYork(_ date: Date) -> String {
        ChronosStyle.day(date, timeZone: "America/New_York") + " · " + ChronosStyle.time(date) + " · America/New_York"
    }

    private func failure(_ error: ChronosDemoFailure) -> String {
        if error == .persistence { return "The check could not be saved. Free some space and retry." }
        if error == .invalidData {
            return "This record or response failed validation. The original forecast has not changed."
        }
        return ChronosCopy.failure(error)
    }

    private func status(_ status: ChronosCheckStatus) -> String {
        switch status {
        case .notChecked: "Recorded · Not checked"
        case .publicationPending: "Publication pending"
        case .partial: "Partly compared"
        case .compared: "Fully compared"
        case .incomparable: "Incomparable New York time"
        case .unverified: "Not independently verified"
        }
    }
}
