public enum IOSDemoLaunch: Equatable, Sendable {
    case models
    case imageSearchEvaluation
    case extractionEvaluation
    case modelLibrarySelfTest
    case modelLibraryAcquisition(String)
    case legacyDemos

    public static func resolve(arguments: [String]) -> Self {
        if arguments.contains("--image-search-eval") { return .imageSearchEvaluation }
        if value("--extract-eval", in: arguments) != nil { return .extractionEvaluation }
        if value("--model-library-selftest", in: arguments) != nil {
            return value("--acquire", in: arguments).map(Self.modelLibraryAcquisition) ?? .modelLibrarySelfTest
        }
        if arguments.contains(where: {
            ["--image-search", "--object-finder", "--truck-inspection", "--legacy-demos"].contains($0)
        }) {
            return .legacyDemos
        }
        return .models
    }

    private static func value(_ name: String, in arguments: [String]) -> String? {
        if let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) {
            return arguments[index + 1]
        }
        return arguments.first { $0.hasPrefix(name + "=") }.map { String($0.dropFirst(name.count + 1)) }
    }
}
