import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryUI

@MainActor
struct ChronosClassificationTests {
    private func model(category: String = "Time Series Forecasting", architecture: String? = "Chronos-T5")
        -> LibraryModel
    {
        LibraryModel(
            id: "chronos2", name: "Chronos-2", category: category, architecture: architecture,
            artifacts: [LibraryFixture().artifact("fp32")], defaultArtifactID: "fp32", platforms: [.iOS],
            demos: [ModelDemoLink(demoID: "text", platforms: [.iOS])])
    }

    @Test("The classification resolves only the category in the requested locale", arguments: ["en_US", "ja_JP"])
    func localizedCategory(_ identifier: String) {
        let model = model()
        let locale = Locale(identifier: identifier)
        var keys: [String] = []
        let classification = ModelClassificationPresentation.classification(for: model, locale: locale) { key, value in
            keys.append(key)
            #expect(value == locale)
            return "[\(value.identifier)] Forecasting"
        }
        #expect(keys == ["Time Series Forecasting"])
        #expect(classification == "[\(identifier)] Forecasting · Chronos-T5")
        #expect(model.classification == "Time Series Forecasting · Chronos-T5")
        #expect(model.category == "Time Series Forecasting")
        #expect(model.architecture == "Chronos-T5")
        #expect(model.name == "Chronos-2")
        #expect(model.artifacts.first?.title == "FP32")
    }

    @Test(
        "Localized rows preserve the selected variant and explicit locale byte formatting",
        arguments: [("en_US", "4 bytes"), ("de_DE", "4 Byte")])
    func localizedRow(_ identifier: String, _ bytes: String) throws {
        let fixture = LibraryFixture()
        let model = model()
        let catalog = try ModelCatalog(models: [model], demos: [fixture.demo("text")])
        let state = fixture.state(catalog)
        let locale = Locale(identifier: identifier)
        let classification = ModelClassificationPresentation.classification(for: model, locale: locale) { _, _ in
            "Localized forecast"
        }
        #expect(
            state.rowDetail(for: model, classification: classification, locale: locale)
                == "Localized forecast · Chronos-T5 · FP32 · \(bytes)")
        #expect(state.rowDetail(for: model, locale: locale) == "Time Series Forecasting · Chronos-T5 · FP32 · \(bytes)")
        #expect(state.catalog == catalog)
        #expect(state.selectedArtifact(for: model.id)?.id == "fp32")
    }

    @Test("An untranslated category falls back to its key and keeps the architecture", arguments: ["en", "ja"])
    func missingTranslation(_ identifier: String) {
        let model = model(category: "Untranslated fixture category")
        #expect(
            ModelClassificationPresentation.classification(for: model, locale: Locale(identifier: identifier))
                == "Untranslated fixture category · Chronos-T5")
    }

    @Test("A bundle resolves categories in the explicit language on every presentation")
    func bundleLanguages() throws {
        let root = URL.temporaryDirectory.appending(path: UUID().uuidString + ".bundle")
        defer { try? FileManager.default.removeItem(at: root) }
        let translations = [("en", "English forecast"), ("ja", "Japanese forecast")]
        for (code, text) in translations {
            let folder = root.appending(path: code + ".lproj")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let data = try PropertyListSerialization.data(
                fromPropertyList: ["Time Series Forecasting": text], format: .xml, options: 0)
            try data.write(to: folder.appending(path: "Localizable.strings"))
        }
        let info = try PropertyListSerialization.data(
            fromPropertyList: ["CFBundleDevelopmentRegion": "en", "CFBundleIdentifier": "fixture.chronos"],
            format: .xml, options: 0)
        try info.write(to: root.appending(path: "Info.plist"))
        let bundle = try #require(Bundle(url: root))
        for (code, text) in translations + translations.reversed() {
            let locale = Locale(identifier: code)
            #expect(
                ModelClassificationPresentation.classification(for: model(), locale: locale, bundle: bundle)
                    == text + " · Chronos-T5")
            #expect(
                ModelClassificationPresentation.classification(
                    for: model(category: "Unknown category"), locale: locale, bundle: bundle)
                    == "Unknown category · Chronos-T5")
        }
    }

    @Test("Empty classification parts keep their existing separator behavior", arguments: [nil, "", "Chronos-T5"])
    func emptyCategory(_ architecture: String?) {
        let model = model(category: "", architecture: architecture)
        let classification = ModelClassificationPresentation.classification(
            for: model, locale: Locale(identifier: "en")
        ) { _, _ in
            Issue.record("An empty category must not be resolved.")
            return "Unexpected translation"
        }
        #expect(classification == (architecture ?? ""))
        #expect(classification == model.classification)
    }
}
