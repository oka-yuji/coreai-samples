import ExtractFeature
import FeatureStore
import Foundation
import GLiNERCore
import Testing

typealias Commands = [Command<ExtractEffect, ExtractFeature.EffectID>]

/// A form in English: one number field and one list field.
let testForm = FormSchema(fields: [
    FormField(key: "pressure", name: "pressure", kind: .number(unit: "MPa")),
    FormField(key: "noise", name: "noise", kind: .list(options: [FormOption(name: "ok"), FormOption(name: "ng")])),
])

let timing = ExtractTiming(preprocess: 1, encode: 2, pair: 3, classify: 4, classifications: 1, map: 0, total: 10)

/// The candidate filter a new state starts with: rules R1 and N1.
let defaultFilter = CandidateFilter.clauseFieldNameAndEquipmentNumber

/// A mapping of `text` with one candidate for each field, at `pressure` and `noise` (Unicode scalar ranges).
func makeMapping(_ text: String, pressure: Range<Int>, noise: Range<Int>, best: String = "ok") -> FormMapping {
    let scalars = Array(text.unicodeScalars)
    func span(_ label: Int, _ range: Range<Int>) -> ExtractedSpan {
        ExtractedSpan(
            labelIndex: label, scalarRange: range, text: String(String.UnicodeScalarView(scalars[range])),
            probability: 0.9)
    }
    let pressureSpan = span(0, pressure)
    return FormMapping(
        fields: [
            FieldMapping(
                field: testForm.fields[0],
                candidates: [
                    FieldCandidate(
                        span: pressureSpan, clause: pressure, number: ParsedNumber(parsing: pressureSpan.text),
                        suggestions: [])
                ]),
            FieldMapping(
                field: testForm.fields[1],
                candidates: [
                    FieldCandidate(
                        span: span(1, noise), clause: noise, number: nil,
                        suggestions: [OptionSuggestion(option: best, probability: 0.8)])
                ]),
        ],
        corrections: [])
}

/// Reduces the actions one by one and returns the commands of each.
func send(_ state: inout ExtractState, _ actions: ExtractAction...) -> [Commands] {
    actions.map { ExtractFeature.reduce(&state, $0) }
}

/// A state in the form mode with an open asset and no text.
func readyState() -> ExtractState {
    var state = ExtractState(form: testForm)
    state.mode = .form
    let info = AssetInfo(path: "/asset", revision: "r1", loadSeconds: 1, floatType: "float32")
    _ = send(&state, .assetFolderChosen(path: "/asset"), .assetOpened(info))
    return state
}

struct ExtractFeatureTests {
    @Test("The initial Extract state and a FormMapper made without a filter use rule R1+N1")
    func defaultsToRulesR1AndN1() {
        #expect(ExtractState(form: testForm).candidateFilter == .clauseFieldNameAndEquipmentNumber)
        #expect(FormMapper(schema: testForm, correctionWords: []).candidateFilter == .clauseFieldNameAndEquipmentNumber)
    }

    @Test(
        "reduce, given a text change, an extraction start, an older version's result and the newer version's result in that order, drops the old result and keeps only the new one"
    )
    func keepsOnlyTheNewestVersion() {
        var state = readyState()
        let first = "pressure 0.4MPa, noise ok"
        let second = "pressure 0.5MPa, noise ng"
        let started = send(
            &state, .textChanged(first), .extractRequested, .textChanged(second), .extractRequested)
        #expect(
            started == [
                [.run(.settle, id: .extraction)],
                [.run(.extract(version: 1, text: first, form: testForm, filter: defaultFilter), id: .extraction)],
                [.run(.settle, id: .extraction)],
                [.run(.extract(version: 2, text: second, form: testForm, filter: defaultFilter), id: .extraction)],
            ])
        #expect(state.phase == .extracting(version: 2))

        let before = state
        let old = makeMapping(first, pressure: 9..<15, noise: 23..<25)
        let stale = send(
            &state, .extractionFinished(version: 1, mapping: old, timing: timing),
            .extractionFailed(version: 1, message: "late"))
        #expect(stale == [[], []])
        #expect(state == before)

        let new = makeMapping(second, pressure: 9..<15, noise: 23..<25, best: "ng")
        _ = send(&state, .extractionFinished(version: 2, mapping: new, timing: timing))
        #expect(state.result == ExtractResult(version: 2, mapping: new, timing: timing, assetRevision: "r1"))
        #expect(state.phase == .ready)
    }

    @Test(
        "reduce clears people's confirmations when the text, the fields or the candidate filter change",
        arguments: [
            ExtractAction.textChanged("pressure 0.5MPa"),
            ExtractAction.fieldsChanged(FormSchema(fields: [testForm.fields[0]])),
            ExtractAction.candidateFilterChanged(.none),
        ])
    func clearsConfirmationsOnANewVersion(change: ExtractAction) {
        var state = readyState()
        let text = "pressure 0.4MPa, noise ok"
        let mapping = makeMapping(text, pressure: 9..<15, noise: 23..<25)
        _ = send(
            &state, .textChanged(text), .extractRequested,
            .extractionFinished(version: 1, mapping: mapping, timing: timing),
            .confirm(field: "pressure", choice: .value(text: "0.4MPa")))
        #expect(state.confirmations == ["pressure": .value(text: "0.4MPa")])

        let commands = ExtractFeature.reduce(&state, change)
        #expect(commands == [.run(.settle, id: .extraction)])
        #expect(state.confirmations.isEmpty)
        #expect(state.result == nil)
        #expect(state.version == 2)
    }

    @Test("reduce keeps a confirmed field's choice when a new result of the same text and fields arrives")
    func keepsConfirmationsAcrossResultsOfOneVersion() {
        var state = readyState()
        let text = "pressure 0.4MPa, noise ok"
        _ = send(
            &state, .textChanged(text), .extractRequested,
            .extractionFinished(
                version: 1, mapping: makeMapping(text, pressure: 9..<15, noise: 23..<25), timing: timing),
            .confirm(field: "noise", choice: .option(name: "ok")), .confirm(field: "pressure", choice: .hold))

        let reopened = AssetInfo(path: "/other", revision: "r2", loadSeconds: 2, floatType: "float32")
        let rerun = send(&state, .assetFolderChosen(path: "/other"), .assetOpened(reopened))
        #expect(
            rerun.last == [
                .run(.extract(version: 1, text: text, form: testForm, filter: defaultFilter), id: .extraction)
            ])
        let changed = makeMapping(text, pressure: 9..<12, noise: 17..<25, best: "ng")
        _ = send(&state, .extractionFinished(version: 1, mapping: changed, timing: timing))

        #expect(state.result?.mapping == changed)
        #expect(state.confirmations == ["noise": .option(name: "ok"), "pressure": .hold])
        _ = send(&state, .unconfirm(field: "pressure"), .confirm(field: "missing", choice: .noMatch))
        #expect(state.confirmations == ["noise": .option(name: "ok")])
    }

    @Test("reduce waits for an open asset and a non-blank text before it extracts, and drops an older folder's result")
    func extractsOnlyWithAnAssetAndText() {
        var state = ExtractState(form: testForm)
        state.mode = .form
        let idle = send(&state, .textChanged("pressure 0.4MPa"), .extractRequested)
        #expect(idle == [[.run(.settle, id: .extraction)], [.cancel(.extraction)]])
        #expect(state.phase == .idle)

        let info = AssetInfo(path: "/second", revision: nil, loadSeconds: 1, floatType: "float32")
        let opened = send(
            &state, .assetFolderChosen(path: "/first"), .assetFolderChosen(path: "/second"),
            .assetFailed(path: "/first", message: "old"), .assetOpened(info))
        #expect(opened[0] == [.run(.openAsset(path: "/first"), id: .asset)])
        #expect(opened[2] == [])
        #expect(
            opened[3] == [
                .run(
                    .extract(version: 1, text: "pressure 0.4MPa", form: testForm, filter: defaultFilter),
                    id: .extraction)
            ])
        #expect(state.asset == .ready(info))

        let blank = send(&state, .textChanged("  \n"), .extractRequested)
        #expect(blank[1] == [.cancel(.extraction)])
        #expect(state.phase == .idle)
    }

    @Test(
        "The export keeps the input, the form, the model output, the rule's marks and people's choices apart, with positions in Unicode scalars and UTF-16"
    )
    func exportsEachSectionApart() throws {
        var state = readyState()
        // A character outside the BMP before the values shifts UTF-16 offsets by one.
        let text = "\u{1F527} pressure 0.4MPa, noise ok"
        let mapping = makeMapping(text, pressure: 11..<17, noise: 25..<27)
        _ = send(
            &state, .textChanged(text), .extractRequested,
            .extractionFinished(version: 1, mapping: mapping, timing: timing),
            .confirm(field: "noise", choice: .option(name: "ok")))
        let commands = ExtractFeature.reduce(&state, .exportRequested)
        #expect(commands.count == 1)
        let document = try #require(
            commands.first.flatMap { command -> ExportDocument? in
                if case .run(.export(let document), id: nil) = command { document } else { nil }
            })
        #expect(document.input.text == text)
        #expect(document.input.version == 1)
        #expect(document.form == testForm)
        #expect(document.confirmations.map(\.choice) == [.option(name: "ok")])
        let model = try #require(document.model)
        #expect(model.probabilitiesCalibrated == false)
        let pressure = try #require(model.fields.first?.candidates.first)
        #expect([pressure.position.scalars.start, pressure.position.scalars.end] == [11, 17] as [Int])
        #expect([pressure.position.utf16.start, pressure.position.utf16.end] == [12, 18] as [Int])
        #expect(pressure.number == "0.4")
        #expect(pressure.unit == "MPa")
        #expect(model.fields.last?.candidates.first?.suggestions.map(\.option) == ["ok"])
        #expect(document.rules?.correctionCues.isEmpty == true)

        let data = try JSONEncoder().encode(document)
        #expect(try JSONDecoder().decode(ExportDocument.self, from: data) == document)
    }

    @Test("The export writes the candidates the filter dropped apart from the candidates people see")
    func exportsTheDroppedCandidates() throws {
        var state = readyState()
        let text = "pressure 0.4MPa, noise ok"
        var mapping = makeMapping(text, pressure: 9..<15, noise: 23..<25)
        mapping.fields[1].droppedCandidates = mapping.fields[1].candidates
        mapping.fields[1].candidates = []
        _ = send(
            &state, .textChanged(text), .extractRequested,
            .extractionFinished(version: 1, mapping: mapping, timing: timing))
        let fields = try #require(ExportDocument(state).model?.fields)
        #expect(fields.map(\.state) == ["candidate", "notDetected"])
        #expect(fields.map(\.candidates.count) == [1, 0])
        #expect(fields.map { $0.droppedCandidates.map(\.text) } == [[], ["ok"]])
    }

    @Test("A field draft keeps the model wording of a field whose name stays, and a new field gets a free key")
    func makesFieldsFromDrafts() {
        let tuned = FormField(
            key: "noise", name: "noise", modelLabel: "noise level",
            kind: .list(
                options: [FormOption(name: "ok", modelLabel: "fine"), FormOption(name: "ng")], modelTask: "noise state")
        )
        let form = FormSchema(fields: [tuned, FormField(key: "field1", name: "x", kind: .number(unit: ""))])

        var edited = FieldDraft(field: tuned)
        edited.options = "ok, ng\u{3001}unchecked"
        edited.description = " heard at the pump "
        #expect(
            edited.field(in: form)
                == FormField(
                    key: "noise", name: "noise", modelLabel: "noise level",
                    kind: .list(
                        options: [
                            FormOption(name: "ok", modelLabel: "fine"), FormOption(name: "ng"),
                            FormOption(name: "unchecked"),
                        ],
                        modelTask: "noise state"),
                    description: "heard at the pump"))

        edited.name = "sound"
        #expect(
            edited.field(in: form)?.kind
                == .list(
                    options: [
                        FormOption(name: "ok", modelLabel: "fine"), FormOption(name: "ng"),
                        FormOption(name: "unchecked"),
                    ], modelTask: nil))

        var added = FieldDraft()
        added.name = "humidity"
        added.unit = "%"
        #expect(added.field(in: form) == FormField(key: "field2", name: "humidity", kind: .number(unit: "%")))
        added.name = " "
        #expect(added.field(in: form) == nil)
        added.name = "state"
        added.kind = .list
        #expect(added.field(in: form) == nil)
    }

    @Test("reduce shows the failure of the current extraction and of the folder being opened, and the export's outcome")
    func showsFailuresAndTheExportOutcome() {
        var state = readyState()
        _ = send(
            &state, .textChanged("pressure 0.4MPa"), .extractRequested, .extractionFailed(version: 1, message: "boom"))
        #expect(state.phase == .failed("boom"))
        _ = send(&state, .assetFolderChosen(path: "/bad"), .assetFailed(path: "/bad", message: "no model"))
        #expect(state.asset == .failed(path: "/bad", message: "no model"))
        _ = send(&state, .exportFinished(path: "/out.json"))
        #expect(state.export == .saved(path: "/out.json"))
        _ = send(&state, .exportFailed("disk full"))
        #expect(state.export == .failed("disk full"))
    }

    @Test("reduce ignores a text or a form that did not change, and forgets a selected field that was removed")
    func ignoresNoChangeAndForgetsARemovedSelection() {
        var state = readyState()
        _ = send(&state, .textChanged("pressure 0.4MPa"), .select(field: "noise"))
        let before = state
        #expect(
            send(
                &state, .textChanged("pressure 0.4MPa"), .fieldsChanged(testForm),
                .candidateFilterChanged(defaultFilter)) == [[], [], []])
        #expect(state == before)
        _ = send(&state, .fieldsChanged(FormSchema(fields: [testForm.fields[0]])))
        #expect(state.selectedField == nil)
        #expect(state.version == 2)
    }

    @Test("reduce opens the last asset folder when the screen appears with no asset open, and only then")
    func reopensTheLastAssetFolder() {
        var state = ExtractState(form: testForm, lastAssetPath: "/last")
        #expect(ExtractFeature.reduce(&state, .appeared) == [.run(.openAsset(path: "/last"), id: .asset)])
        #expect(state.asset == .opening(path: "/last"))
        #expect(ExtractFeature.reduce(&state, .appeared) == [])
        var fresh = ExtractState(form: testForm)
        #expect(ExtractFeature.reduce(&fresh, .appeared) == [])
        #expect(fresh.asset == .none)
    }

    @Test(
        "reduce starts no extraction while an input method converts a word, stops the one under way, and waits again for the text when the conversion ends without a result"
    )
    func waitsForTheConversionToEnd() {
        var state = readyState()
        let text = "pressure 0.4MPa, noise ok"
        let extract = Command<ExtractEffect, ExtractFeature.EffectID>.run(
            .extract(version: 1, text: text, form: testForm, filter: defaultFilter), id: .extraction)
        let steps = send(
            &state, .textChanged(text), .compositionChanged(true), .extractRequested, .compositionChanged(true),
            .compositionChanged(false), .extractRequested, .compositionChanged(true))
        #expect(
            steps == [
                [.run(.settle, id: .extraction)], [.cancel(.extraction)], [.cancel(.extraction)], [],
                [.run(.settle, id: .extraction)], [extract], [.cancel(.extraction)],
            ])
        #expect(state.phase == .idle)

        let mapping = makeMapping(text, pressure: 9..<15, noise: 23..<25)
        let finished = send(
            &state, .compositionChanged(false), .extractRequested,
            .extractionFinished(version: 1, mapping: mapping, timing: timing), .compositionChanged(true),
            .compositionChanged(false))
        #expect(finished == [[.run(.settle, id: .extraction)], [extract], [], [.cancel(.extraction)], []])
        #expect(state.result?.mapping == mapping)
        #expect(state.phase == .ready)
    }

    @Test("A correction cue belongs to the fields with a candidate in its sentence")
    func showsCuesInTheSentenceOfACandidate() {
        var state = readyState()
        let text = "pressure 0.4MPa, sorry, 0.5MPa\u{3002}noise ok"
        var mapping = makeMapping(text, pressure: 24..<30, noise: 37..<39)
        mapping.corrections = [CorrectionCue(range: 17..<22, word: "sorry")]
        _ = send(
            &state, .textChanged(text), .extractRequested,
            .extractionFinished(version: 1, mapping: mapping, timing: timing))
        #expect(state.cues(forField: "pressure") == mapping.corrections)
        #expect(state.cues(forField: "noise").isEmpty)
    }

    @Test(
        "reduce carries neither the confirmations nor the result across a switch between the label mode and the form mode"
    )
    func switchingModesStartsOver() {
        #expect(ExtractState(form: testForm).mode == .labels)
        var state = readyState()
        let labels = FormSchema(fields: [FormField(key: "value", name: "value", kind: .number(unit: ""))])
        state.labels = labels
        let text = "pressure 0.4MPa, noise ok"
        let mapping = makeMapping(text, pressure: 9..<15, noise: 23..<25)
        _ = send(
            &state, .textChanged(text), .extractRequested,
            .extractionFinished(version: 1, mapping: mapping, timing: timing), .select(field: "noise"),
            .confirm(field: "noise", choice: .option(name: "ok")))
        #expect(state.confirmations == ["noise": .option(name: "ok")])

        let toLabels = send(
            &state, .modeChanged(.labels), .extractionFinished(version: 1, mapping: mapping, timing: timing),
            .extractRequested, .confirm(field: "noise", choice: .option(name: "ok")))
        #expect(
            toLabels == [
                [.run(.settle, id: .extraction)], [],
                [.run(.extract(version: 2, text: text, form: labels, filter: .none), id: .extraction)], [],
            ])
        #expect(state.result == nil)
        #expect(state.confirmations.isEmpty)
        #expect(state.selectedField == nil)
        let labelMapping = FormMapping(fields: [FieldMapping(field: labels.fields[0], candidates: [])], corrections: [])
        _ = send(&state, .extractionFinished(version: 2, mapping: labelMapping, timing: timing))
        #expect(state.result?.mapping == labelMapping)

        let toForm = send(&state, .modeChanged(.form), .extractRequested)
        #expect(
            toForm == [
                [.run(.settle, id: .extraction)],
                [.run(.extract(version: 3, text: text, form: testForm, filter: defaultFilter), id: .extraction)],
            ])
        #expect(state.result == nil)
        #expect(state.confirmations.isEmpty)
        #expect(send(&state, .modeChanged(.form)) == [[]])
    }

    @Test(
        "In the label mode, reduce extracts nothing without a label, edits and extracts the labels and not the form with no candidate filter, and the export writes the labels and no filter"
    )
    func labelModeWorksOnTheLabels() {
        var state = ExtractState(form: testForm)
        _ = send(
            &state, .assetFolderChosen(path: "/asset"),
            .assetOpened(AssetInfo(path: "/asset", revision: nil, loadSeconds: 1, floatType: "float32")))
        let text = "pressure 0.4MPa, noise ok"
        #expect(
            send(&state, .textChanged(text), .extractRequested) == [
                [.run(.settle, id: .extraction)], [.cancel(.extraction)],
            ])
        #expect(state.phase == .idle)

        let labels = FormSchema(fields: [FormField(key: "value", name: "value", kind: .number(unit: ""))])
        let added = send(&state, .fieldsChanged(labels), .extractRequested)
        #expect(
            added == [
                [.run(.settle, id: .extraction)],
                [.run(.extract(version: 2, text: text, form: labels, filter: .none), id: .extraction)],
            ])
        #expect(state.labels == labels)
        #expect(state.form == testForm)
        #expect(send(&state, .fieldsChanged(labels)) == [[]])

        let mapping = FormMapping(fields: [FieldMapping(field: labels.fields[0], candidates: [])], corrections: [])
        _ = send(&state, .extractionFinished(version: 2, mapping: mapping, timing: timing))
        let document = ExportDocument(state)
        #expect(document.mode == "labels")
        #expect(document.form == labels)
        #expect(document.model?.fields.map(\.key) == ["value"])
        #expect(document.model?.candidateFilter == "none")
        #expect(document.confirmations.isEmpty)
    }

    @Test(
        "The exported JSON carries the candidate filter's name and the asset's revision, and a filter change starts over"
    )
    func exportsTheFilterAndTheRevision() throws {
        var state = readyState()
        state.appVersion = "1.0 (1)"
        let text = "pressure 0.4MPa, noise ok"
        let mapping = makeMapping(text, pressure: 9..<15, noise: 23..<25)
        _ = send(
            &state, .textChanged(text), .extractRequested,
            .extractionFinished(version: 1, mapping: mapping, timing: timing))
        let json = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(ExportDocument(state))) as? [String: Any])
        let model = try #require(json["model"] as? [String: Any])
        #expect(json["mode"] as? String == "form")
        #expect(model["candidateFilter"] as? String == "clauseFieldNameAndEquipmentNumber")
        #expect(model["assetRevision"] as? String == "r1")
        #expect(model["appVersion"] as? String == "1.0 (1)")

        let off = send(&state, .candidateFilterChanged(.none), .extractRequested)
        #expect(off[1] == [.run(.extract(version: 2, text: text, form: testForm, filter: .none), id: .extraction)])
        #expect(state.result == nil)
        // A result that arrives while another folder opens has no known revision.
        _ = send(
            &state, .assetFolderChosen(path: "/other"),
            .extractionFinished(version: 2, mapping: mapping, timing: timing))
        let output = try #require(ExportDocument(state).model)
        #expect(output.candidateFilter == "none")
        #expect(output.assetRevision == nil)
    }
}
