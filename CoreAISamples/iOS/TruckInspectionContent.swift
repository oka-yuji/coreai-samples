#if os(iOS)
    import FeatureStore
    import SwiftUI
    import TruckInspectionCore
    import TruckInspectionFeature

    struct TruckInspectionContent<Models: View>: View {
        let store: Store<TruckInspectionFeature>
        @ViewBuilder let models: Models
        @Environment(\.locale) private var locale
        @State private var resetting = false
        @State private var proposedProfile: TruckProfile?

        private var state: InspectionState { store.state }

        var body: some View {
            introduction
            configuration
            controls
            if state.phase == .answering || state.phase == .classifying || state.phase == .confirming
                || state.phase == .speaking || state.phase == .listening
            {
                question
            }
            if !state.message.isEmpty {
                Section { Text(LocalizedStringKey(state.message)).foregroundStyle(.secondary) }
            }
            records
            diagnostics
            Section {
                ForEach(state.catalog.sources, id: \.url) { source in
                    Link(destination: source.url) { Text(verbatim: source.title) }
                }
            } header: {
                Text("Checklist sources")
            }
        }

        private var introduction: some View {
            Section {
                Label("Daily inspection results", systemImage: "truck.box.fill")
                    .font(.headline)
                Text(verbatim: state.catalog.notice).font(.caption).foregroundStyle(.secondary)
                ProgressView(value: Double(state.records.count), total: Double(state.items.count))
                Text("\(state.records.count) of \(state.items.count) recorded")
                    .font(.subheadline).monospacedDigit()
            }
        }

        private var configuration: some View {
            Section {
                Picker(
                    "Vehicle profile",
                    selection: Binding(
                        get: { state.profile },
                        set: { profile in
                            if state.records.isEmpty {
                                store.send(.profileChanged(profile))
                            } else if profile != state.profile {
                                proposedProfile = profile
                            }
                        })
                ) {
                    ForEach(TruckProfile.allCases, id: \.self) { profile in
                        Text(verbatim: state.catalog.profiles[profile.rawValue] ?? profile.rawValue).tag(profile)
                    }
                }
                .disabled(!state.canConfigure)
                .confirmationDialog("Clear this inspection record?", isPresented: $resetting) {
                    Button("Start a new record", role: .destructive) { store.send(.reset) }
                }
                .confirmationDialog(
                    "Changing the vehicle profile clears this record.",
                    isPresented: Binding(get: { proposedProfile != nil }, set: { if !$0 { proposedProfile = nil } })
                ) {
                    if let profile = proposedProfile {
                        Button("Change vehicle profile", role: .destructive) {
                            store.send(.profileChanged(profile))
                            proposedProfile = nil
                        }
                    }
                }
                Text("Japanese questions and answers are used with either interface language.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }

        @ViewBuilder private var controls: some View {
            Section {
                if state.phase == .completed {
                    Label("All answers recorded", systemImage: "checkmark.circle")
                    Text("Review the results below, including items needing attention and items not inspected.")
                        .font(.caption)
                } else if state.canConfigure {
                    Button("Start Japanese voice input") { store.send(.voice(.start)) }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .disabled(!state.canStartVoice)
                    Button(state.records.isEmpty ? "Start inspection" : "Continue inspection") { store.send(.start) }
                        .buttonStyle(.bordered).controlSize(.large)
                } else if state.busy {
                    HStack {
                        ProgressView()
                        Text(LocalizedStringKey(phaseTitle))
                    }
                }
                if let limit = state.voiceLimit {
                    Text(LocalizedStringKey(limit.message)).font(.caption).foregroundStyle(.secondary)
                }
                if state.voiceActive {
                    Label(
                        LocalizedStringKey(phaseTitle),
                        systemImage: state.phase == .listening ? "mic.fill" : "speaker.wave.2.fill")
                    if !state.partialText.isEmpty { Text(verbatim: state.partialText).font(.body) }
                    Button("Use answer buttons") { store.send(.voice(.useButtons)) }
                }
                if let failure = state.stopFailureMessage {
                    Text(LocalizedStringKey(failure)).font(.caption).foregroundStyle(.secondary)
                }
                if state.phase != .idle && state.phase != .completed {
                    Button("Stop", role: .cancel) { store.send(.stop) }
                        .disabled(state.phase == .stopping)
                }
                if state.canConfigure && !state.records.isEmpty {
                    Button("New inspection") { resetting = true }
                }
            }
        }

        @ViewBuilder private var question: some View {
            if let item = state.currentItem {
                Section {
                    Text(verbatim: item.group).font(.caption).foregroundStyle(.secondary)
                    Text(verbatim: item.question).font(.title3.weight(.semibold))
                        .fixedSize(horizontal: false, vertical: true)
                    Text(verbatim: item.guidance).font(.caption).foregroundStyle(.secondary)
                    if item.conditionBased {
                        Label("Check timing depends on distance and operating conditions.", systemImage: "calendar")
                            .font(.caption)
                    }
                    if state.phase == .confirming || (state.voiceActive && state.voicePrompt == .confirmation),
                        let pending = state.pending
                    {
                        Text(verbatim: state.draft).font(.body)
                        Text(verbatim: state.catalog.confirmation(for: item, status: pending.status))
                            .font(.headline)
                        HStack {
                            Button("Confirm result") { store.send(.confirm(true)) }.buttonStyle(.borderedProminent)
                            Button("Answer again") { store.send(.confirm(false)) }.buttonStyle(.bordered)
                        }
                        .controlSize(.large)
                        .disabled(state.phase != .confirming)
                    } else {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 130))], spacing: 12) {
                            ForEach(state.catalog.choices, id: \.status) { choice in
                                Button {
                                    store.send(.choose(choice.status))
                                } label: {
                                    Text(verbatim: choice.title).font(.headline)
                                        .frame(maxWidth: .infinity, minHeight: 44)
                                }
                                .buttonStyle(.bordered)
                                .disabled(!state.canAnswer)
                            }
                        }
                        TextField(
                            "Or type what you found",
                            text: Binding(get: { state.draft }, set: { store.send(.draftChanged($0)) }), axis: .vertical
                        )
                        .lineLimit(2...5).disabled(!state.canAnswer)
                        Button("Interpret answer") { store.send(.submit) }
                            .disabled(
                                !state.canAnswer || state.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
        }

        private var records: some View {
            Section {
                ForEach(state.items) { item in
                    VStack(alignment: .leading, spacing: 5) {
                        HStack(alignment: .top) {
                            Text(verbatim: item.title)
                            Spacer()
                            if let record = state.records[item.id] {
                                Text(verbatim: state.catalog.title(for: record.candidate.status))
                                    .fontWeight(.semibold)
                                    .foregroundStyle(
                                        record.candidate.status == .attention ? Color.orange : Color.primary)
                            } else {
                                Text("Unanswered").foregroundStyle(.secondary)
                            }
                        }
                        if let record = state.records[item.id] {
                            Text(verbatim: record.answer).font(.caption).foregroundStyle(.secondary)
                            HStack {
                                Text(LocalizedStringKey(record.confirmed ? "Confirmed" : "Explicit answer"))
                                    .font(.caption)
                                Spacer()
                                Button("Edit") { store.send(.edit(item.id)) }.disabled(!state.canConfigure)
                            }
                        }
                    }
                }
                ShareLink(item: state.report) { Label("Share inspection record", systemImage: "square.and.arrow.up") }
            } header: {
                Text("Inspection record")
            }
        }

        private var diagnostics: some View {
            Section {
                DisclosureGroup("Model and diagnostics") {
                    models
                    if let info = state.modelInfo {
                        Text("Model ready: \(state.artifact ?? info.folder.lastPathComponent)")
                        Text(
                            "Load: \(info.milliseconds.formatted(.number.precision(.fractionLength(0)).locale(locale))) ms"
                        )
                        Text("Compute preference: \(info.computePreference)")
                    } else {
                        Text("Tap answers work without a model.")
                    }
                    Button("Measure 20 classifications") { store.send(.measure) }
                        .disabled(!state.canMeasure)
                    if let measurement = state.measurement {
                        Text(
                            "Median: \(measurement.medianMilliseconds.formatted(.number.precision(.fractionLength(1)).locale(locale))) ms"
                        )
                        Text(
                            "p95: \(measurement.p95Milliseconds.formatted(.number.precision(.fractionLength(1)).locale(locale))) ms"
                        )
                    }
                    Text(LocalizedStringKey(phaseTitle)).font(.caption)
                    if !state.voiceDiagnostics.isEmpty { Text(verbatim: state.voiceDiagnostics).font(.caption) }
                    ForEach(Array(state.events.suffix(8).enumerated()), id: \.offset) { _, event in
                        Text(verbatim: event).font(.caption2).textSelection(.enabled)
                    }
                    ShareLink(item: state.log) { Label("Copy or share diagnostics", systemImage: "doc.on.doc") }
                }
            }
        }

        private var phaseTitle: String {
            switch state.phase {
            case .idle: "Ready"
            case .preparing: "Preparing the model…"
            case .answering: "Waiting for your answer"
            case .classifying: "Interpreting your answer…"
            case .confirming: "Waiting for confirmation"
            case .completed: "All answers recorded"
            case .stopping: "Stopping…"
            case .stopFailed: "Stopping failed"
            case .measuring: "Measuring classification speed…"
            case .authorizing: "Checking voice permissions…"
            case .speaking: "Speaking…"
            case .listening:
                state.voicePrompt == .confirmation ? "Listening for confirmation" : "Listening for your answer"
            }
        }
    }
#endif
