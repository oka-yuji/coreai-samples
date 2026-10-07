import Foundation
import SwiftUI

struct DemoApp: App {
    var body: some Scene {
        WindowGroup("Core AI Demos") {
            DemoRootView()
        }
        .defaultSize(width: 1040, height: 720)
    }
}

@main
enum DemoAppMain {
    static func main() {
        if LaunchArguments.flag("--selftest") {
            SelfTest.run()
        }
        if LaunchArguments.flag("--agent-selftest") {
            AgentSelfTest.run()
        }
        if LaunchArguments.flag("--agent-e2e") {
            AgentSelfTest.runEndToEnd()
        }
        if LaunchArguments.value("--extract-eval") != nil {
            ExtractEval.run()
        }
        DemoApp.main()
    }
}
