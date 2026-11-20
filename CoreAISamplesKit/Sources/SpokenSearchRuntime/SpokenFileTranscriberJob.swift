import AVFAudio
import CoreMedia
import Foundation
import Speech
import SpokenSearchFeature

actor SpokenFileTranscriberJob {
    private var analyzer: SpeechAnalyzer?

    func recognize(_ url: URL) async throws -> SpokenTranscript {
        try Task.checkCancellation()
        guard url.isFileURL, let file = try? AVAudioFile(forReading: url), file.length > 0 else {
            throw SpokenRecognitionError.unreadableFile
        }
        guard SpeechTranscriber.isAvailable else { throw SpokenRecognitionError.unavailable }
        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "ja-JP")) else {
            throw SpokenRecognitionError.languageUnavailable
        }
        try Task.checkCancellation()
        let module = SpeechTranscriber(
            locale: locale, transcriptionOptions: [], reportingOptions: [], attributeOptions: [.audioTimeRange])
        try await prepare(module, locale: locale)
        try Task.checkCancellation()
        let analyzer = SpeechAnalyzer(
            modules: [module], options: .init(priority: .userInitiated, modelRetention: .whileInUse))
        self.analyzer = analyzer
        do {
            try await analyzer.start(inputAudioFile: file, finishAfterFile: true)
            var runs: [SpokenRun] = []
            for try await result in module.results where result.isFinal {
                try Task.checkCancellation()
                for (time, range) in result.text.runs[\.audioTimeRange] {
                    runs.append(
                        SpokenRun(
                            text: String(result.text[range].characters), start: time?.start.seconds,
                            end: time?.end.seconds))
                }
            }
            try Task.checkCancellation()
            self.analyzer = nil
            return SpokenTranscript(runs: runs)
        } catch {
            try Task.checkCancellation()
            throw SpokenRecognitionError.recognitionFailed
        }
    }

    func finish() async {
        await analyzer?.cancelAndFinishNow()
        analyzer = nil
    }

    private func prepare(_ module: SpeechTranscriber, locale: Locale) async throws {
        let status = try await assetStatus(for: module, locale: locale)
        if status == .installed { return }
        guard status == .supported || status == .downloading else { throw SpokenRecognitionError.languageUnavailable }
        do {
            if let request = try await AssetInventory.assetInstallationRequest(supporting: [module]) {
                try await withTaskCancellationHandler {
                    try Task.checkCancellation()
                    try await request.downloadAndInstall()
                } onCancel: {
                    request.progress.cancel()
                }
            }
            try Task.checkCancellation()
            guard try await assetStatus(for: module, locale: locale) == .installed else {
                throw SpokenRecognitionError.languageUnavailable
            }
        } catch {
            try Task.checkCancellation()
            throw SpokenRecognitionError.languageUnavailable
        }
    }

    private func assetStatus(for module: SpeechTranscriber, locale: Locale) async throws -> AssetInventory.Status {
        let status = await AssetInventory.status(forModules: [module])
        try Task.checkCancellation()
        if status == .installed { return status }
        let installed = await SpeechTranscriber.installedLocales
        try Task.checkCancellation()
        return Self.effectiveStatus(status, for: locale, installedLocales: installed)
    }

    static func effectiveStatus(
        _ status: AssetInventory.Status, for locale: Locale, installedLocales: [Locale]
    ) -> AssetInventory.Status {
        let requested = locale.identifier(.bcp47)
        return installedLocales.contains { $0.identifier(.bcp47) == requested } ? .installed : status
    }
}
