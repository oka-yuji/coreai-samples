import FeatureStore
import Foundation
import GLiNERCore

public enum ExtractFeature: Feature {
    public enum EffectID: Hashable, Sendable {
        case extraction
        case asset
    }

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
            guard composing else {
                return state.result == nil && !state.isManaged ? [.run(.settle, id: .extraction)] : []
            }
            if case .extracting = state.phase { state.phase = .idle }
            return [.cancel(.extraction)]
        case .extractRequested:
            return extract(&state)
        case .extractionFinished(let version, let mapping, let timing):
            guard version == state.version else { return [] }

            let revision: String? = if case .ready(let info) = state.asset { info.revision } else { nil }
            state.result = ExtractResult(version: version, mapping: mapping, timing: timing, assetRevision: revision)
            state.resourceIdentity = if case .managed(_, let identity) = state.execution { identity } else { nil }
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
            guard !state.isManaged, state.asset == .none, let path = state.lastAssetPath else { return [] }
            return reduce(&state, .assetFolderChosen(path: path))
        case .assetFolderChosen(let path):
            guard !state.isManaged else { return [] }
            state.asset = .opening(path: path)
            return [.run(.openAsset(path: path), id: .asset)]
        case .assetOpened(let info):
            guard state.asset == .opening(path: info.path) else { return [] }
            state.asset = .ready(info)
            guard !state.isManaged || state.phase == .extracting(version: state.version) else { return [] }
            return extract(&state)
        case .assetFailed(let path, let message):
            guard state.asset == .opening(path: path) else { return [] }
            state.asset = .failed(path: path, message: message)
            if state.isManaged, case .extracting = state.phase { state.phase = .failed(message) }
            return []
        case .closed:
            guard state.isManaged else { return [] }
            if case .extracting = state.phase { state.phase = .idle }
            return [.cancelAll, .run(.release, id: .asset)]
        case .released:
            guard state.isManaged else { return [] }
            state.asset = .none
            return []
        case .releaseFailed(let message):
            guard case .managed(let path, _) = state.execution else { return [] }
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
        state.resourceIdentity = nil
        state.confirmations = [:]
        state.phase = .idle
        return [state.isManaged ? .cancel(.extraction) : .run(.settle, id: .extraction)]
    }

    private static func extract(_ state: inout ExtractState) -> [Command<ExtractEffect, EffectID>] {
        guard !state.composing, !state.schema.fields.isEmpty, !state.text.allSatisfy(\.isWhitespace) else {
            state.phase = .idle
            return [.cancel(.extraction)]
        }
        if case .managed(let path, _) = state.execution {
            switch state.asset {
            case .ready:
                break
            case .opening:
                state.phase = .extracting(version: state.version)
                return []
            case .none, .failed:
                state.asset = .opening(path: path)
                state.phase = .extracting(version: state.version)
                return [.run(.openAsset(path: path), id: .asset)]
            }
        }
        guard case .ready = state.asset else {
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

        case labels

        case form
    }

    public enum Execution: Equatable, Sendable {
        case standalone
        case managed(path: String, identity: String)
    }

    public var text = ""

    public var version = 0
    public var mode = Mode.labels

    public var labels: FormSchema

    public var form: FormSchema

    public var candidateFilter = CandidateFilter.clauseFieldNameAndEquipmentNumber

    public var result: ExtractResult?

    public var confirmations: [String: ExtractChoice] = [:]

    public var selectedField: String?
    public var phase = Phase.idle
    public var asset = Asset.none
    public var export = Export.none

    public var lastAssetPath: String?

    public var composing = false

    public var appVersion: String?

    public var execution = Execution.standalone

    public var resourceIdentity: String?

    public var schema: FormSchema { mode == .labels ? labels : form }

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

    var isManaged: Bool {
        if case .managed = execution { true } else { false }
    }

    public func managed(path: String, identity: String) -> ExtractState {
        var state = self
        state.execution = .managed(path: path, identity: identity)
        state.asset = .none
        state.composing = false
        if case .extracting = state.phase { state.phase = .idle }
        return state
    }

    public func cues(forField key: String) -> [CorrectionCue] {
        guard let mapping = result?.mapping, let field = mapping.fields.first(where: { $0.field.key == key }) else {
            return []
        }
        let scalars = Array(text.unicodeScalars)
        let sentences = field.candidates.map { Sentence.range(around: $0.span.scalarRange, in: scalars) }
        return mapping.corrections.filter { cue in sentences.contains { $0.overlaps(cue.range) } }
    }
}

public enum ExtractAction: Equatable, Sendable {

    case textChanged(String)

    case compositionChanged(Bool)

    case fieldsChanged(FormSchema)
    case modeChanged(ExtractState.Mode)
    case candidateFilterChanged(CandidateFilter)

    case extractRequested
    case extractionFinished(version: Int, mapping: FormMapping, timing: ExtractTiming)
    case extractionFailed(version: Int, message: String)
    case select(field: String?)

    case appeared
    case confirm(field: String, choice: ExtractChoice)
    case unconfirm(field: String)
    case assetFolderChosen(path: String)
    case assetOpened(AssetInfo)
    case assetFailed(path: String, message: String)
    case exportRequested
    case exportFinished(path: String)
    case exportFailed(String)
    case closed
    case released
    case releaseFailed(String)
}

public enum ExtractEffect: Equatable, Sendable {

    case settle

    case openAsset(path: String)

    case extract(version: Int, text: String, form: FormSchema, filter: CandidateFilter)

    case export(ExportDocument)

    case release
}

public struct ExtractResult: Equatable, Sendable {
    public var version: Int
    public var mapping: FormMapping
    public var timing: ExtractTiming

    public var assetRevision: String?

    public init(version: Int, mapping: FormMapping, timing: ExtractTiming, assetRevision: String? = nil) {
        self.version = version
        self.mapping = mapping
        self.timing = timing
        self.assetRevision = assetRevision
    }
}

public enum ExtractChoice: Codable, Hashable, Sendable {

    case option(name: String)

    case value(text: String)

    case noMatch

    case hold
}

public struct ExtractTiming: Codable, Equatable, Sendable {
    public var preprocess: Double
    public var encode: Double

    public var pair: Double

    public var classify: Double
    public var classifications: Int

    public var map: Double

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

    public var revision: String?
    public var loadSeconds: Double

    public var floatType: String

    public init(path: String, revision: String?, loadSeconds: Double, floatType: String) {
        self.path = path
        self.revision = revision
        self.loadSeconds = loadSeconds
        self.floatType = floatType
    }
}

enum Sentence {

    static let ends: Set<Unicode.Scalar> = ["\u{3002}", "\n"]

    static let fullStop: Set<Unicode.Scalar> = ["\u{3002}"]

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
