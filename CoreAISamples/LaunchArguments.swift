import Foundation

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
