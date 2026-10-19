#if os(iOS)
    import ModelLibraryFeature
    import SwiftUI
    import UIKit

    @main
    struct PhoneDemoApp: App {
        var body: some Scene {
            WindowGroup {
                switch IOSDemoLaunch.resolve(arguments: CommandLine.arguments) {
                case .models:
                    PhoneModelLibraryScreen()
                case .imageSearchEvaluation:
                    PhoneImageSearchEvalView()
                case .extractionEvaluation:
                    PhoneEvalView()
                case .modelLibrarySelfTest:
                    ProgressView().task { exit(await IOSModelLibraryAssembly.runSelfTest()) }
                case .modelLibraryAcquisition(let target):
                    ProgressView().task {
                        UIApplication.shared.isIdleTimerDisabled = true
                        exit(await IOSModelLibraryAssembly.runSelfTest(acquire: target))
                    }
                case .legacyDemos:
                    PhoneLegacyDemosView()
                }
            }
        }
    }
#endif
