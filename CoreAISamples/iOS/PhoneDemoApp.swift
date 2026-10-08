#if os(iOS)
    import ExtractFeature
    import FeatureStore
    import Foundation
    import SwiftUI
    import UIKit

    @main
    struct PhoneDemoApp: App {
        private let computeUnits: ComputeUnits

        private let assetFolders: [URL]
        @State private var extract: Store<ExtractFeature>

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
                if LaunchArguments.value("--extract-eval") != nil {
                    PhoneEvalView()
                } else {
                    PhoneExtractView(store: extract, assetFolders: assetFolders, computeUnits: computeUnits)
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
