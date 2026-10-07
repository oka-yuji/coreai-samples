import CoreAIBackend
import Foundation

enum DecideCommand {
    static func run() async {
        do {
            let arguments = CommandLine.arguments
            guard let index = arguments.firstIndex(of: "--decide-json"), index + 2 < arguments.count,
                let path = LaunchArguments.value("--decision-bundle")
                    ?? ProcessInfo.processInfo.environment["COREAI_DECISION_BUNDLE"]
            else {
                throw DecisionError.invalidRequest(
                    "Usage: CoreAISamples --decide-json input.json output.json --decision-bundle folder")
            }
            let request = try DecisionRequest(json: Data(contentsOf: URL(filePath: arguments[index + 1])))
            let model = try await DecisionModel.load(bundle: URL(filePath: path))
            let result = try await model.decide(request)
            try result.systemOneJSON(for: request)
                .write(to: URL(filePath: arguments[index + 2]), options: .atomic)
        } catch {
            FileHandle.standardError.write(Data(("Decide failed: \(error.localizedDescription)\n").utf8))
            exit(1)
        }
    }
}
