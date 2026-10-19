import Observation

@MainActor
@Observable
final class DemoNavigator {
    var selection: Demo? =
        CommandLine.arguments.contains("--image-search")
        ? .imageSearch : CommandLine.arguments.contains("--video-search") ? .videoSearch : .chat
}
