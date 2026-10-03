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
        DemoApp.main()
    }
}

enum LaunchArguments {
    static func value(_ name: String) -> String? {
        let arguments = CommandLine.arguments
        if let index = arguments.firstIndex(of: name), index + 1 < arguments.count { return arguments[index + 1] }
        return arguments.first { $0.hasPrefix(name + "=") }.map { String($0.dropFirst(name.count + 1)) }
    }

    static func flag(_ name: String) -> Bool {
        CommandLine.arguments.contains(name)
    }
}
