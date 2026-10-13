import Observation

@MainActor
@Observable
final class DemoNavigator {
    var selection: Demo? = .chat
}
