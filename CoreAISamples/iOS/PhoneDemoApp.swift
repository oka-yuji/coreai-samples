#if os(iOS)
    import ModelLibraryFeature
    import SwiftUI

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
                case .legacyDemos:
                    PhoneLegacyDemosView()
                }
            }
        }
    }
#endif
