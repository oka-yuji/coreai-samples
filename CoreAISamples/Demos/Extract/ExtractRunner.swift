import ExtractFeature
import Foundation
import GLiNERCore
import GLiNERRuntime

/// Runs the effects of `ExtractFeature` for its store. Every failure comes back as an action; a cancelled effect
/// sends nothing.
@MainActor
struct ExtractRunner {
    /// The last asset folder that opened, in `UserDefaults`.
    static let assetFolderKey = "extractAssetFolder"
    nonisolated static var exportsFolder: URL {
        ModelStorage.root.appending(path: "extract-exports", directoryHint: .isDirectory)
    }
    /// The app's version and build, as `metrics.jsonl` writes them, for the export and `--extract-eval`.
    nonisolated static var appVersion: String {
        func value(_ key: String) -> String { Bundle.main.infoDictionary?[key] as? String ?? "?" }
        return "\(value("CFBundleShortVersionString")) (\(value("CFBundleVersion")))"
    }
    /// The note the Extract screens of the Mac and the iPhone show.
    static let notice =
        "Technical verification of a Core AI conversion. Zero-shot GLiNER2.5-multi did not reach the pre-registered "
        + "bar for Japanese inspection memos (sealed set: fabricated candidates 9/199, primary-span precision 0.791, "
        + "recall 0.874). Suggestions are uncalibrated scores."

    let service: any ExtractService

    func run(_ effect: ExtractEffect, send: (ExtractAction) -> Void) async {
        switch effect {
        case .settle:
            do {
                try await Task.sleep(for: ExtractFeature.settleDelay)
            } catch {
                return
            }
            send(.extractRequested)
        case .openAsset(let path):
            do {
                var info = try await service.open(folder: URL(filePath: path, directoryHint: .isDirectory))
                // The reducer matches the result to the request by this path; a directory URL's path ends in "/".
                info.path = path
                UserDefaults.standard.set(path, forKey: Self.assetFolderKey)
                send(.assetOpened(info))
            } catch is CancellationError {
                return
            } catch {
                send(.assetFailed(path: path, message: Self.describe(error)))
            }
        case .extract(let version, let text, let form, let filter):
            do {
                let (mapping, timing) = try await service.extract(text: text, schema: form, filter: filter)
                send(.extractionFinished(version: version, mapping: mapping, timing: timing))
            } catch is CancellationError {
                return
            } catch {
                send(.extractionFailed(version: version, message: Self.describe(error)))
            }
        case .export(let document):
            do {
                send(.exportFinished(path: try await Self.write(document)))
            } catch {
                send(.exportFailed(Self.describe(error)))
            }
        }
    }

    /// Writes `document` to a new file in `exportsFolder`, off the main actor, and returns its path. The name carries
    /// the time to the millisecond, so a second export does not replace the first.
    @concurrent
    static func write(_ document: ExportDocument) async throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(document)
        try FileManager.default.createDirectory(at: exportsFolder, withIntermediateDirectories: true)
        let stamp = Date.now.formatted(
            .iso8601.year().month().day().time(includingFractionalSeconds: true).timeZone(separator: .omitted))
        let url = exportsFolder.appending(path: "extract-\(stamp.replacing(":", with: "")).json")
        try data.write(to: url, options: .atomic)
        return url.path(percentEncoded: false)
    }

    static func describe(_ error: any Error) -> String {
        switch error {
        case GLiNERPreprocessingError.inputTooLong(let tokens, let limit):
            "The note is too long for the model: \(tokens) tokens, over the limit of \(limit)."
        case GLiNERPreprocessingError.tooManyLabels(let count, let limit):
            "Too many labels: \(count), over the limit of \(limit)."
        case GLiNERAssetError.modelCount(let count):
            "The folder must hold exactly one .aimodel; it holds \(count)."
        case GLiNERAssetError.contractMismatch(let part):
            "contract.json does not match what this app implements (\(part))."
        case GLiNERModelError.missingFunction(let name):
            "The model has no function \(name)."
        case GLiNERModelError.signatureMismatch(let function, let value):
            "\(function) of the model differs from what this app implements at \(value): it takes and returns Int32 "
                + "and Float32 (FP32) values of the contract's shapes."
        case GLiNERModelError.unreadableOutput(let name):
            "The model returned \(name) as something other than a contiguous Float32 array."
        default:
            error.localizedDescription
        }
    }

    static func describe(_ timing: ExtractTiming) -> String {
        func milliseconds(_ value: Double) -> String { value.formatted(.number.precision(.fractionLength(0))) + " ms" }
        return "preprocess \(milliseconds(timing.preprocess)) · encode \(milliseconds(timing.encode)) · "
            + "pair \(milliseconds(timing.pair)) · classify \(timing.classifications)× \(milliseconds(timing.classify)) · "
            + "map \(milliseconds(timing.map)) · total \(milliseconds(timing.total))"
    }
}

/// The bundled examples: form U1, the correction words, example notes and labels, in Japanese, kept out of the source.
enum ExtractResources {
    struct Examples: Decodable {
        let notes: [String]
        /// The labels the label mode starts with.
        let labels: FormSchema
        /// A field the screen check adds to the form.
        let extraField: FormField
    }

    static let form = load(FormSchema.self, "inspection_u1")
    static let correctionWords = load([String].self, "correction_words")
    static let examples = load(Examples.self, "extract_examples")

    /// A JSON file of the app bundle. The files ship with the app, so one that is missing or broken is a build
    /// error: stop rather than run with an empty form or no correction words.
    private static func load<Value: Decodable>(_ type: Value.Type, _ name: String) -> Value {
        do {
            guard let url = Bundle.main.url(forResource: name, withExtension: "json") else {
                throw CocoaError(.fileNoSuchFile)
            }
            return try JSONDecoder().decode(type, from: Data(contentsOf: url))
        } catch {
            preconditionFailure("The bundled \(name).json cannot be read: \(error)")
        }
    }
}
