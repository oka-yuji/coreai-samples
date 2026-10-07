import CryptoKit
import ExtractFeature
import Foundation
import GLiNERCore

/// `--extract-eval <gold.json> --asset <folder> --out <result.json> [--form <form.json>]
/// [--compute-units gpu|default] [--candidate-filter r1|r1n1|none]`: maps every sentence of a gold file that is not
/// excluded with the same service as the screen, scores the mappings and writes the report.
enum ExtractEval {
    private static let usage =
        "usage: CoreAISamples --extract-eval <gold.json> --asset <folder> --out <result.json> [--form <form.json>] "
        + "[--compute-units gpu|default] [--candidate-filter r1|r1n1|none]"

    struct Output: Encodable {
        struct Asset: Encodable {
            let path: String
            /// The type of the model's floating-point inputs and outputs: `float32` or `float16`.
            let floatType: String
        }

        /// A candidate the rule dropped. Positions count Unicode scalars of the sentence.
        struct Dropped: Encodable {
            /// The sentence's `id`.
            let id: String
            let field: String
            let start: Int
            let end: Int
            let text: String
            let probability: Float

            /// Every candidate the rule dropped, in sentence, form and text order.
            static func all(in scored: [(gold: GoldSentence, mapping: FormMapping)]) -> [Dropped] {
                scored.flatMap { sentence, mapping in
                    mapping.fields.flatMap { field in
                        field.droppedCandidates.map {
                            Dropped(
                                id: sentence.id, field: field.field.key, start: $0.span.scalarRange.lowerBound,
                                end: $0.span.scalarRange.upperBound, text: $0.span.text,
                                probability: $0.span.probability)
                        }
                    }
                }
            }
        }

        /// The hardware model identifier: `hw.machine` on iOS, `hw.model` on macOS.
        let device: String
        /// The OS version with its build, as `ProcessInfo.operatingSystemVersionString` gives it.
        let osVersion: String
        /// `gpu` or `default`, from `--compute-units`.
        let computeUnits: String
        let gold: String
        let goldSHA256: String
        /// `bundled` or the path of `--form`.
        let form: String
        let asset: Asset
        let revision: String?
        /// The rule of `--candidate-filter`: `clauseFieldNameAndEquipmentNumber` (`r1n1`, rules R1 and N1, the default
        /// and the screen's), `clauseFieldName` (`r1`, rule R1) or `none`.
        let candidateFilter: String
        /// The report does not score them.
        let droppedCandidates: [Dropped]
        let appVersion: String
        let assetLoadSeconds: Double
        let sentences: Int
        let excluded: [String]
        /// From the start of preprocessing to the end of the mapping, in milliseconds. `cold` is a first run on the
        /// first sentence, before the measured runs.
        let timing: TimingSummary
        let report: EvaluationReport
    }

    static func run() -> Never {
        // swiftlint:disable:next task_only_in_store - The headless entry point has no store; it runs once and exits.
        Task { @MainActor in exit(await execute()) }
        RunLoop.main.run()
        fatalError("the main run loop returned")
    }

    @MainActor
    static func execute() async -> Int32 {
        guard let goldPath = LaunchArguments.value("--extract-eval"), let assetPath = LaunchArguments.value("--asset"),
            let outPath = LaunchArguments.value("--out"), let computeUnits = ComputeUnits.launchValue,
            let filter = candidateFilter
        else {
            log(usage)
            return 2
        }
        do {
            let goldData = try Data(contentsOf: URL(filePath: goldPath))
            let gold = try GoldSentence.decode(goldData)
            let formPath = LaunchArguments.value("--form")
            let form =
                try formPath.map {
                    try JSONDecoder().decode(FormSchema.self, from: Data(contentsOf: URL(filePath: $0)))
                }
                ?? ExtractResources.form
            let sentences = gold.filter { $0.excluded != true }
            guard !form.fields.isEmpty, let first = sentences.first else {
                log("[extract-eval] nothing to score: \(form.fields.count) fields, \(sentences.count) sentences")
                return 1
            }
            let service = GLiNERExtractService(
                correctionWords: ExtractResources.correctionWords, computeUnits: computeUnits)
            let asset = try await service.open(folder: URL(filePath: assetPath, directoryHint: .isDirectory))
            log(
                "[extract-eval] asset \(asset.path) revision \(asset.revision ?? "none") loaded in \(asset.loadSeconds) s"
                    + " floatType \(asset.floatType) computeUnits \(computeUnits.rawValue) device \(hardwareModel)"
            )
            let cold = try await service.extract(text: first.text, schema: form, filter: filter).timing
            var scored: [(gold: GoldSentence, mapping: FormMapping)] = []
            var warm: [ExtractTiming] = []
            for (index, sentence) in sentences.enumerated() {
                let (mapping, timing) = try await service.extract(text: sentence.text, schema: form, filter: filter)
                scored.append((sentence, mapping))
                warm.append(timing)
                log("[extract-eval] \(index + 1)/\(sentences.count) \(sentence.id) \(ExtractRunner.describe(timing))")
            }
            let output = Output(
                device: hardwareModel, osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
                computeUnits: computeUnits.rawValue,
                gold: goldPath, goldSHA256: SHA256.hash(data: goldData).map { String(format: "%02x", $0) }.joined(),
                form: formPath ?? "bundled", asset: Output.Asset(path: asset.path, floatType: asset.floatType),
                revision: asset.revision,
                candidateFilter: filter.name, droppedCandidates: Output.Dropped.all(in: scored),
                appVersion: ExtractRunner.appVersion,
                assetLoadSeconds: asset.loadSeconds, sentences: sentences.count,
                excluded: gold.filter { $0.excluded == true }.map(\.id),
                timing: ExtractEvaluation.timing(cold: cold, warm: warm), report: ExtractEvaluation.evaluate(scored))
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(output).write(to: URL(filePath: outPath), options: .atomic)
            report(output)
            FileHandle.standardOutput.write(Data("EXTRACT EVAL DONE \(outPath)\n".utf8))
            return 0
        } catch {
            log("[extract-eval] failed: \(ExtractRunner.describe(error))")
            return 1
        }
    }

    /// `--candidate-filter`: rules R1 and N1 without one, or nil for a value it does not know.
    private static var candidateFilter: CandidateFilter? {
        switch LaunchArguments.value("--candidate-filter") ?? "r1n1" {
        case "r1": .clauseFieldName
        case "r1n1": .clauseFieldNameAndEquipmentNumber
        case "none": CandidateFilter.none
        default: nil
        }
    }

    /// The overall and per-field scores, one line each, on stdout.
    private static func report(_ output: Output) {
        var lines = [
            "[extract-eval] sentences=\(output.sentences) excluded=\(output.excluded.count) revision=\(output.revision ?? "none")"
        ]
        lines.append("overall " + describe(output.report.overall))
        for (key, summary) in output.report.fields.sorted(by: { $0.key < $1.key }) {
            lines.append("field \(key) " + describe(summary))
        }
        let timing = output.timing
        lines.append(
            "timing warm p50=\(format(timing.p50?.total))ms p95=\(format(timing.p95?.total))ms "
                + "cold=\(format(timing.cold?.total))ms runs=\(timing.runs) assetLoad=\(format(output.assetLoadSeconds))s"
        )
        FileHandle.standardOutput.write(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    private static func describe(_ summary: Summary) -> String {
        let mention = summary.mention
        var parts = [
            "mention P=\(format(mention.precision)) R=\(format(mention.recall)) F1=\(format(mention.f1))",
            "found=\(mention.found) fabricated=\(ratio(summary.fabrication)) main=\(ratio(summary.mainFabrication))",
            "missed=\(ratio(summary.miss)) misplaced=\(mention.misplaced)",
            "option=\(ratio(summary.option)) withCandidate=\(ratio(summary.optionWithCandidate))",
            "top2=\(ratio(summary.optionTopTwo)) nullOption=\(ratio(summary.optionNull))",
            "number=\(ratio(summary.number)) inCandidates=\(ratio(summary.numberInCandidates))",
            "noDigits=\(summary.numberWithoutDigits)",
            "spanP=\(ratio(summary.spans.matched)) spanR=\(ratio(summary.spans.found))",
            "exactP=\(ratio(summary.spans.exactMatched)) exactR=\(ratio(summary.spans.exactFound))",
            "agreement=\(ratio(summary.agreement))",
        ]
        if let all = summary.allFields { parts.append("allFields=\(ratio(all))") }
        return parts.joined(separator: " ")
    }

    private static func ratio(_ ratio: Ratio) -> String { "\(ratio.count)/\(ratio.total)" }

    private static func format(_ value: Double?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(3)).grouping(.never)) } ?? "n/a"
    }

    private static func log(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    /// `hw.machine` on iOS; on macOS that is the CPU architecture, and `hw.model` names the Mac.
    private static var hardwareModel: String {
        #if os(iOS)
            let name = "hw.machine"
        #else
            let name = "hw.model"
        #endif
        var size = 0
        sysctlbyname(name, nil, &size, nil, 0)
        var value = [UInt8](repeating: 0, count: max(size, 1))
        sysctlbyname(name, &value, &size, nil, 0)
        return String(decoding: value.prefix { $0 != 0 }, as: UTF8.self)
    }
}
