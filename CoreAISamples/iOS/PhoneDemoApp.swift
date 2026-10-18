#if os(iOS)
    import ExtractFeature
    import FeatureStore
    import Foundation
    import ImageSearchFeature
    import ObjectFinderFeature
    import ProductFeature
    import SwiftUI
    import UIKit
    import TruckInspectionFeature

    @main
    struct PhoneDemoApp: App {
        private let computeUnits: ComputeUnits

        private let assetFolders: [URL]
        private let truckAssetFolders: [URL]
        @State private var extract: Store<ExtractFeature>
        @State private var products = ProductAssembly.makeStore()
        @State private var imageSearch = ImageSearchAssembly.makeStore()
        @State private var showingImageSearch = LaunchArguments.flag("--image-search")
        @State private var objectFinder = ObjectFinderAssembly.makeSession()
        @State private var showingObjectFinder = LaunchArguments.flag("--object-finder")
        @State private var showingExtractionLab = false
        @State private var showingTruckInspection = CommandLine.arguments.contains("--truck-inspection")
        @State private var truck: Result<Store<TruckInspectionFeature>, any Error>
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
            let folders = TruckInspectionAssembly.modelFolders(assetFolders)
            truckAssetFolders = folders
            _truck = State(initialValue: Result { try TruckInspectionAssembly.makeStore(assetFolders: folders) })
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
                                Menu {
                                    Section("GLiNER2.5-multi") {
                                        Button("Truck inspection") { showingTruckInspection = true }
                                        Button("Extraction lab") { showingExtractionLab = true }
                                    }
                                    Section("EmbeddingGemma 2") {
                                        Button("Image search") { showingImageSearch = true }
                                        Button("Find objects with the camera") { showingObjectFinder = true }
                                    }
                                } label: {
                                    Label("Demos", systemImage: "square.grid.2x2")
                                }
                                .labelStyle(.iconOnly)
                            }
                            .sheet(isPresented: $showingExtractionLab) {
                                PhoneExtractView(store: extract, assetFolders: assetFolders, computeUnits: computeUnits)
                            }
                            .sheet(isPresented: $showingTruckInspection) {
                                switch truck {
                                case .success(let store):
                                    TruckInspectionView(store: store, assetFolders: truckAssetFolders)
                                case .failure(let error):
                                    ContentUnavailableView(
                                        "Could not load the inspection questions",
                                        systemImage: "exclamationmark.triangle",
                                        description: Text(verbatim: String(describing: error)))
                                }
                            }
                            .sheet(isPresented: $showingImageSearch) {
                                NavigationStack {
                                    ImageSearchView(store: imageSearch)
                                        .toolbar { Button("Done") { showingImageSearch = false } }
                                }
                            }
                            .sheet(isPresented: $showingObjectFinder) {
                                NavigationStack {
                                    ObjectFinderView(store: objectFinder.store, preview: objectFinder.preview)
                                        .toolbar { Button("Done") { showingObjectFinder = false } }
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
                            folder: documents.appending(path: ImageSearchAssembly.preparedFolderName)))
                }
        }
    }
#endif
