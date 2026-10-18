public enum IOSDemoLaunch: Equatable, Sendable {
    case models
    case imageSearchEvaluation
    case extractionEvaluation
    case modelLibrarySelfTest
    case legacyDemos

    public static func resolve(arguments: [String]) -> Self {
        if arguments.contains("--image-search-eval") { return .imageSearchEvaluation }
        if let index = arguments.firstIndex(of: "--extract-eval"), arguments.indices.contains(index + 1) {
            return .extractionEvaluation
        }
        if arguments.contains(where: { $0.hasPrefix("--extract-eval=") }) { return .extractionEvaluation }
        if let index = arguments.firstIndex(of: "--model-library-selftest"), arguments.indices.contains(index + 1) {
            return .modelLibrarySelfTest
        }
        if arguments.contains(where: { $0.hasPrefix("--model-library-selftest=") }) { return .modelLibrarySelfTest }
        if arguments.contains(where: {
            ["--image-search", "--object-finder", "--truck-inspection", "--legacy-demos"].contains($0)
        }) {
            return .legacyDemos
        }
        return .models
    }
}
