import Observation

@MainActor
@Observable
final class DemoNavigator {
    var selection: Demo? = CommandLine.arguments.contains("--image-search") ? .imageSearch : .chat
}
