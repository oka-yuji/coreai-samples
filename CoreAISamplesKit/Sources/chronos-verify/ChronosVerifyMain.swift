import ChronosRuntime
import Darwin
import Foundation

@main
struct ChronosVerifyMain {
    static func main() async {
        do {
            let command = try ChronosCommand.parse(Array(CommandLine.arguments.dropFirst()))
            if command == .help {
                print(ChronosCommand.usage)
                return
            }
            try await ChronosCommand.execute(command)
        } catch {
            let message = "chronos-verify failed: \(error)\n"
            do {
                try FileHandle.standardError.write(contentsOf: Data(message.utf8))
            } catch {
                exit(1)
            }
            exit(1)
        }
    }
}
