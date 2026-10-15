#if os(iOS)
    import ExtractFeature
    import FeatureStore
    import Foundation
    import ProductFeature
    import SwiftUI
    import UIKit
    import TruckInspectionFeature

    @main
    struct PhoneDemoApp: App {
        private let computeUnits: ComputeUnits

        private let assetFolders: [URL]
        @State private var extract: Store<ExtractFeature>
        @State private var products = ProductAssembly.makeStore()
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
            let folders = assetFolders
            _truck = State(initialValue: Result { try TruckInspectionAssembly.makeStore(assetFolders: folders) })
        }

        var body: some Scene {
            WindowGroup {
                if LaunchArguments.value("--extract-eval") != nil {
                    PhoneEvalView()
                } else {
                    NavigationStack {
                        ProductComparisonView(store: products, assetFolders: assetFolders)
                            .toolbar {
                                DemoLanguagePicker(code: $languageCode)
                                Button("Truck inspection") { showingTruckInspection = true }
                                Button("Extraction lab") { showingExtractionLab = true }
                            }
                            .sheet(isPresented: $showingExtractionLab) {
                                PhoneExtractView(store: extract, assetFolders: assetFolders, computeUnits: computeUnits)
                            }
                            .sheet(isPresented: $showingTruckInspection) {
                                switch truck {
                                case .success(let store):
                                    TruckInspectionView(store: store, assetFolders: assetFolders)
                                case .failure(let error):
                                    ContentUnavailableView(
                                        "Could not load the inspection questions",
                                        systemImage: "exclamationmark.triangle",
                                        description: Text(verbatim: String(describing: error)))
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
#endif
