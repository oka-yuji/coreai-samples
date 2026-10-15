#if os(iOS)
    import ExtractFeature
    import FeatureStore
    import Foundation
    import ImageSearchFeature
    import ProductFeature
    import SwiftUI
    import UIKit

    @main
    struct PhoneDemoApp: App {
        private let computeUnits: ComputeUnits

        private let assetFolders: [URL]
        @State private var extract: Store<ExtractFeature>
        @State private var products = ProductAssembly.makeStore()
        @State private var imageSearch = ImageSearchAssembly.makeStore()
        @State private var showingImageSearch = LaunchArguments.flag("--image-search")
        @State private var showingExtractionLab = false
        @AppStorage(DemoLanguage.preferenceKey) private var languageCode = DemoLanguage.initialCode

        init() {
            let documents = URL.documentsDirectory
            FileManager.default.changeCurrentDirectoryPath(documents.path(percentEncoded: false))
            let items =
                (try? FileManager.default.contentsOfDirectory(at: documents, includingPropertiesForKeys: nil)) ?? []
            assetFolders = items.filter {
                FileManager.default.fileExists(atPath: $0.appending(path: "contract.json").path(percentEncoded: false))
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            let computeUnits = ComputeUnits.launchValue ?? .gpu
            self.computeUnits = computeUnits
            let service = GLiNERExtractService(
                correctionWords: ExtractResources.correctionWords, computeUnits: computeUnits)
            _extract = State(
                initialValue: Store(
                    initialState: ExtractState(form: ExtractResources.form, labels: ExtractResources.examples.labels),
                    run: ExtractRunner(service: service).run))
        }

        var body: some Scene {
            WindowGroup {
                if LaunchArguments.flag("--image-search-eval") {
                    PhoneImageSearchEvalView()
                } else if LaunchArguments.value("--extract-eval") != nil {
                    PhoneEvalView()
                } else {
                    NavigationStack {
                        ProductComparisonView(store: products, assetFolders: assetFolders)
                            .toolbar {
                                DemoLanguagePicker(code: $languageCode)
                                Button("Extraction lab") { showingExtractionLab = true }
                                Button("Image search") { showingImageSearch = true }
                            }
                            .sheet(isPresented: $showingExtractionLab) {
                                PhoneExtractView(store: extract, assetFolders: assetFolders, computeUnits: computeUnits)
                            }
                            .sheet(isPresented: $showingImageSearch) {
                                NavigationStack {
                                    ImageSearchView(store: imageSearch)
                                        .toolbar { Button("Done") { showingImageSearch = false } }
                                }
                            }
                    }
                    .environment(\.locale, (DemoLanguage(rawValue: languageCode) ?? .english).locale)
                }
            }
        }
    }

    struct PhoneEvalView: View {
        var body: some View {
            ProgressView("Running --extract-eval…")
                .task {
                    UIApplication.shared.isIdleTimerDisabled = true
                    exit(await ExtractEval.execute())
                }
        }
    }

    struct PhoneImageSearchEvalView: View {
        var body: some View {
            ProgressView("Checking image search…")
                .task {
                    UIApplication.shared.isIdleTimerDisabled = true
                    let documents = URL.documentsDirectory
                    let run = LaunchArguments.value("--image-search-eval") ?? "latest"
                    exit(
                        await ImageSearchSelfTest.run(
                            directory: documents.appending(path: "embeddinggemma2-eval-" + run),
                            folder: documents.appending(path: "embeddinggemma2")))
                }
        }
    }
#endif
