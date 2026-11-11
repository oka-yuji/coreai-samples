import ChronosFeature
import FeatureStore
import SwiftUI

struct ChronosAPIControls: View {
    let store: Store<ChronosFeature>
    private var state: ChronosFeature.State { store.state }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker(selection: Binding(get: { state.source }, set: { store.send(.selectSource($0)) })) {
                Text("Fixed scenes", bundle: .module).tag(ChronosFeature.Source.fixed)
                Text("Official NYC API dates", bundle: .module).tag(ChronosFeature.Source.api)
            } label: {
                Text("Data source", bundle: .module)
            }
            .pickerStyle(.segmented)
            if state.source == .api {
                Text(
                    "The NYC API publishes historical annual datasets. TLC monthly files are a separate publication. A recent catalog update does not mean recent trip data.",
                    bundle: .module
                )
                .font(.caption).foregroundStyle(.secondary)
                Button {
                    store.send(.refreshCoverage)
                } label: {
                    Text("Check published dates", bundle: .module)
                }
                .disabled(state.phase != .ready)
                if let coverage = state.coverage {
                    Text(
                        "Published years: \(coverage.datasets.map { String($0.year) }.joined(separator: ", "))",
                        bundle: .module
                    )
                    .font(.caption)
                    if let date = state.requestedOrigin, let range = coverage.selectableRange {
                        DatePicker(
                            selection: Binding(
                                get: { date },
                                set: { day in
                                    if let slot = NYCTime.matchingSlot(on: day, preserving: date, in: range),
                                        coverage.allowsHistory(endingAt: slot)
                                    {
                                        store.send(.selectDate(slot))
                                    } else if let slot = NYCTime.slots(on: day).first(where: {
                                        coverage.allowsHistory(endingAt: $0)
                                    }) {
                                        store.send(.selectDate(slot))
                                    }
                                }), in: range, displayedComponents: .date
                        ) { Text("New York date", bundle: .module) }
                        .environment(\.timeZone, NYCTime.calendar.timeZone)
                        .environment(\.calendar, NYCTime.calendar)
                        Picker(selection: Binding(get: { date }, set: { store.send(.selectDate($0)) })) {
                            ForEach(NYCTime.slots(on: date).filter { coverage.allowsHistory(endingAt: $0) }, id: \.self)
                            { slot in
                                Text(slotLabel(slot)).tag(slot)
                            }
                        } label: {
                            Text("New York time", bundle: .module)
                        }
                        .pickerStyle(.menu)
                    }
                    if coverage.selectableRange == nil {
                        Text("No published history is available.", bundle: .module)
                    }
                }
                HStack {
                    Button {
                        store.send(.fetchHistory)
                    } label: {
                        Text("Fetch 21-day history", bundle: .module)
                    }
                    .disabled(!state.canFetchHistory)
                    Button {
                        store.send(.fetchActuals)
                    } label: {
                        Text("Fetch actual pickups", bundle: .module)
                    }
                    .disabled(!state.canFetchActuals)
                }
                Text(
                    "Fetch history, choose on-device prediction, then fetch actual pickups when you want to compare. Missing counts stay missing; published zero counts stay zero.",
                    bundle: .module
                )
                .font(.caption).foregroundStyle(.secondary)
                if !state.liveAvailable {
                    Text(
                        "API dates require the prepared model for prediction. Fixed scenes remain available without it.",
                        bundle: .module)
                }
                Text(
                    LocalizedStringKey(
                        state.catalog?.isAPI == true ? "Displayed result: NYC API" : "Displayed result: fixed scene"),
                    bundle: .module
                )
                .font(.caption.bold())
                if let history = state.selectedScenario?.apiHistory {
                    let missing = history.availability.filter { $0 != .observed }.count
                    Text("Missing history intervals: \(missing)", bundle: .module).font(.caption)
                }
            }
        }
    }

    private func slotLabel(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = NYCTime.calendar.timeZone
        formatter.dateFormat = "HH:mm zzz"
        return formatter.string(from: date)
    }
}
