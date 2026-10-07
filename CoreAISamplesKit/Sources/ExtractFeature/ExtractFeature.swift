import FeatureStore
import Foundation
import GLiNERCore

/// Extract: a note, the labels people type (or, as an experiment, a form), the model's candidates for each, and the
/// values people confirm in the form.
///
/// Every change to the text, the labels or form being extracted, the mode or the candidate filter starts a new
/// version. A result carries the version it was asked for and is kept only while that version is current. A new
/// version clears people's confirmations; a result of the same version replaces the candidates and keeps them.
public enum ExtractFeature: Feature {
    public enum EffectID: Hashable, Sendable {
        case extraction
        case asset
    }

    /// The pause after the last change before the note is extracted.
    public static let settleDelay: Duration = .milliseconds(300)

    public static func reduce(_ state: inout ExtractState, _ action: ExtractAction) -> [Command<
        ExtractEffect, EffectID
    >] {
        switch action {
        case .textChanged(let text):
            guard text != state.text else { return [] }
            state.text = text
            return startVersion(&state)
        case .fieldsChanged(let form):
            guard form != state.schema else { return [] }
            if state.mode == .labels { state.labels = form } else { state.form = form }
            if let selected = state.selectedField, !form.fields.contains(where: { $0.key == selected }) {
                state.selectedField = nil
            }
            return startVersion(&state)
        case .modeChanged(let mode):
            guard mode != state.mode else { return [] }
            state.mode = mode
            state.selectedField = nil
            return startVersion(&state)
        case .candidateFilterChanged(let filter):
            guard filter != state.candidateFilter else { return [] }
            state.candidateFilter = filter
            return startVersion(&state)
        case .compositionChanged(let composing):
            guard composing != state.composing else { return [] }
            state.composing = composing
            guard composing else { return state.result == nil ? [.run(.settle, id: .extraction)] : [] }
            if case .extracting = state.phase { state.phase = .idle }
            return [.cancel(.extraction)]
        case .extractRequested:
            return extract(&state)
        case .extractionFinished(let version, let mapping, let timing):
            guard version == state.version else { return [] }
            // A result that arrives while another folder opens came from the folder before it: its revision is unknown.
            let revision: String? = if case .ready(let info) = state.asset { info.revision } else { nil }
            state.result = ExtractResult(version: version, mapping: mapping, timing: timing, assetRevision: revision)
            state.phase = .ready
            return []
        case .extractionFailed(let version, let message):
            guard version == state.version else { return [] }
            state.phase = .failed(message)
            return []
        case .select(let key):
            state.selectedField = key
            return []
        case .confirm(let key, let choice):
            guard state.mode == .form, state.form.fields.contains(where: { $0.key == key }) else { return [] }
            state.confirmations[key] = choice
            return []
        case .unconfirm(let key):
            state.confirmations[key] = nil
            return []
        case .appeared:
            guard state.asset == .none, let path = state.lastAssetPath else { return [] }
            return reduce(&state, .assetFolderChosen(path: path))
        case .assetFolderChosen(let path):
            state.asset = .opening(path: path)
            return [.run(.openAsset(path: path), id: .asset)]
        case .assetOpened(let info):
            guard state.asset == .opening(path: info.path) else { return [] }
            state.asset = .ready(info)
            return extract(&state)
        case .assetFailed(let path, let message):
            guard state.asset == .opening(path: path) else { return [] }
            state.asset = .failed(path: path, message: message)
            return []
        case .exportRequested:
            return [.run(.export(ExportDocument(state)), id: nil)]
        case .exportFinished(let path):
            state.export = .saved(path: path)
            return []
        case .exportFailed(let message):
            state.export = .failed(message)
            return []
        }
    }

    private static func startVersion(_ state: inout ExtractState) -> [Command<ExtractEffect, EffectID>] {
        state.version += 1
        state.result = nil
        state.confirmations = [:]
        state.phase = .idle
        return [.run(.settle, id: .extraction)]
    }

    private static func extract(_ state: inout ExtractState) -> [Command<ExtractEffect, EffectID>] {
        guard case .ready = state.asset, !state.composing, !state.schema.fields.isEmpty,
            !state.text.allSatisfy(\.isWhitespace)
        else {
            state.phase = .idle
            return [.cancel(.extraction)]
        }
        state.phase = .extracting(version: state.version)
        let effect = ExtractEffect.extract(
            version: state.version, text: state.text, form: state.schema, filter: state.appliedFilter)
        return [.run(effect, id: .extraction)]
    }
}

public struct ExtractState: Equatable, Sendable {
    public enum Phase: Equatable, Sendable {
        /// Nothing to extract yet: no text, no fields, no asset, a word being converted, or the text still settling.
        case idle
        case extracting(version: Int)
        case ready
        case failed(String)
    }

    public enum Asset: Equatable, Sendable {
        case none
        case opening(path: String)
        case ready(AssetInfo)
        case failed(path: String, message: String)
    }

    public enum Export: Equatable, Sendable {
        case none
        case saved(path: String)
        case failed(String)
    }

    public enum Mode: String, Sendable {
        /// People type labels and see the spans the model returns for them. Nothing is confirmed.
        case labels
        /// The experiment: an inspection form with suggestions for its list fields, and people's choices.
        case form
    }

    public var text = ""
    /// The version of what is extracted: every change of the text, the labels or form, the mode or the filter adds one.
    public var version = 0
    public var mode = Mode.labels
    /// The labels of the label mode, as number fields without a unit: no clause is classified.
    public var labels: FormSchema
    /// The form of the experiment.
    public var form: FormSchema
    /// The rule `FormMapper` drops the form's candidates by: rules R1 and N1 unless people choose another.
    public var candidateFilter = CandidateFilter.clauseFieldNameAndEquipmentNumber
    /// The latest result of the current version; nil until one arrives.
    public var result: ExtractResult?
    /// People's choices in the form, by field key.
    public var confirmations: [String: ExtractChoice] = [:]
    /// The field whose evidence is emphasized in the note.
    public var selectedField: String?
    public var phase = Phase.idle
    public var asset = Asset.none
    public var export = Export.none
    /// The asset folder opened last time; `appeared` opens it again when no asset is open.
    public var lastAssetPath: String?
    /// An input method is converting a word in the note: nothing is extracted until it is committed.
    public var composing = false
    /// The app's version and build, for the export.
    public var appVersion: String?

    /// The labels or the form, as the mode says: what is extracted.
    public var schema: FormSchema { mode == .labels ? labels : form }
    /// The rule applied to what is extracted, as the mode says: none for the labels, which are number fields whose
    /// spans on equipment names rule N1 would drop, and `candidateFilter` for the form.
    public var appliedFilter: CandidateFilter { mode == .labels ? .none : candidateFilter }

    public init(
        form: FormSchema, labels: FormSchema = FormSchema(fields: []), lastAssetPath: String? = nil,
        appVersion: String? = nil
    ) {
        self.form = form
        self.labels = labels
        self.lastAssetPath = lastAssetPath
        self.appVersion = appVersion
    }

    /// The correction cues in the sentences that hold a candidate of the field. They only mark the place.
    public func cues(forField key: String) -> [CorrectionCue] {
        guard let mapping = result?.mapping, let field = mapping.fields.first(where: { $0.field.key == key }) else {
            return []
        }
        let scalars = Array(text.unicodeScalars)
        let sentences = field.candidates.map { Sentence.range(around: $0.span.scalarRange, in: scalars) }
        return mapping.corrections.filter { cue in sentences.contains { $0.overlaps(cue.range) } }
    }
}

public enum ExtractAction: Sendable {
    /// The committed text of the note.
    case textChanged(String)
    /// An input method started (true) or stopped (false) converting a word in the note.
    case compositionChanged(Bool)
    /// The labels in the label mode, the form in the form mode.
    case fieldsChanged(FormSchema)
    case modeChanged(ExtractState.Mode)
    case candidateFilterChanged(CandidateFilter)
    /// The text has settled: extract it when an asset is open.
    case extractRequested
    case extractionFinished(version: Int, mapping: FormMapping, timing: ExtractTiming)
    case extractionFailed(version: Int, message: String)
    case select(field: String?)
    /// The screen appeared.
    case appeared
    case confirm(field: String, choice: ExtractChoice)
    case unconfirm(field: String)
    case assetFolderChosen(path: String)
    case assetOpened(AssetInfo)
    case assetFailed(path: String, message: String)
    case exportRequested
    case exportFinished(path: String)
    case exportFailed(String)
}

public enum ExtractEffect: Equatable, Sendable {
    /// Wait `ExtractFeature.settleDelay`, then send `extractRequested`.
    case settle
    /// Open the asset folder, then send `assetOpened` or `assetFailed`.
    case openAsset(path: String)
    /// Extract and map the text with the filter, then send `extractionFinished` or `extractionFailed` with the version.
    case extract(version: Int, text: String, form: FormSchema, filter: CandidateFilter)
    /// Write the document as JSON, then send `exportFinished` or `exportFailed`.
    case export(ExportDocument)
}

public struct ExtractResult: Equatable, Sendable {
    public var version: Int
    public var mapping: FormMapping
    public var timing: ExtractTiming
    /// The `revision` of the asset that was open when the result arrived; nil when none was.
    public var assetRevision: String?

    public init(version: Int, mapping: FormMapping, timing: ExtractTiming, assetRevision: String? = nil) {
        self.version = version
        self.mapping = mapping
        self.timing = timing
        self.assetRevision = assetRevision
    }
}

/// What a person chose for a field.
public enum ExtractChoice: Codable, Hashable, Sendable {
    /// An option of a list field, by name.
    case option(name: String)
    /// A candidate of a number field, as written in the note.
    case value(text: String)
    /// None of the options or candidates fits.
    case noMatch
    /// Left undecided for now.
    case hold
}

/// The time of one extraction, in milliseconds.
public struct ExtractTiming: Codable, Equatable, Sendable {
    public var preprocess: Double
    public var encode: Double
    /// The candidate pool, `pair` and decoding.
    public var pair: Double
    /// Every clause classification, each with its own preprocessing and encoding.
    public var classify: Double
    public var classifications: Int
    /// `FormMapper.map`.
    public var map: Double
    /// From the start of preprocessing to the end of the mapping.
    public var total: Double

    public init(
        preprocess: Double, encode: Double, pair: Double, classify: Double, classifications: Int, map: Double,
        total: Double
    ) {
        self.preprocess = preprocess
        self.encode = encode
        self.pair = pair
        self.classify = classify
        self.classifications = classifications
        self.map = map
        self.total = total
    }
}

public struct AssetInfo: Equatable, Sendable {
    public var path: String
    /// The `revision` in the asset's `contract.json`.
    public var revision: String?
    public var loadSeconds: Double
    /// The type of the model's floating-point inputs and outputs: `float32` or `float16`.
    public var floatType: String

    public init(path: String, revision: String?, loadSeconds: Double, floatType: String) {
        self.path = path
        self.revision = revision
        self.loadSeconds = loadSeconds
        self.floatType = floatType
    }
}

/// Sentences of a note: they end at one of `ends`, which belongs to the sentence.
enum Sentence {
    /// The ideographic full stop and the line break, for the screen.
    static let ends: Set<Unicode.Scalar> = ["\u{3002}", "\n"]
    /// The ideographic full stop only, as the scoring rules define a sentence.
    static let fullStop: Set<Unicode.Scalar> = ["\u{3002}"]

    /// From the start of the sentence that holds the start of `range` to the end of the sentence that holds its end.
    static func range(
        around range: Range<Int>, in scalars: [Unicode.Scalar], ends: Set<Unicode.Scalar> = Sentence.ends
    ) -> Range<Int> {
        var lower = min(range.lowerBound, scalars.count)
        while lower > 0, !ends.contains(scalars[lower - 1]) { lower -= 1 }
        var upper = min(max(range.upperBound, lower), scalars.count)
        if upper > lower, ends.contains(scalars[upper - 1]) { return lower..<upper }
        while upper < scalars.count, !ends.contains(scalars[upper]) { upper += 1 }
        return lower..<min(upper + 1, scalars.count)
    }
}
