import ArgumentParser
import Foundation

struct RunFunction: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "function",
        abstract:
            "Run a function of a Core AI asset that has no state on inputs read from files, and write its outputs to files and one record."
    )

    @Option(help: "The .aimodel folder of the asset.")
    var model: String

    @Option(help: "Name of the function to run.")
    var function: String

    @Option(
        help:
            "JSON file that lists the inputs with name, dtype, shape and file. Each file is relative to the folder of this JSON file."
    )
    var inputs: String

    @Option(help: "Output folder. It must not exist yet.")
    var output: String

    @Option(
        name: .customLong("repeat"),
        help: "Number of calls with the same inputs. The outputs of every call must equal those of the first call.")
    var repeatCount = 1

    func validate() throws {
        guard repeatCount > 0 else { throw ValidationError("--repeat takes a count of one or more.") }
    }

    func run() async throws {
        try await execute(source: AIModelFunctionSource(url: URL(filePath: model)))
    }

    func execute(source: some FunctionSource) async throws {
        let record: FunctionRecord
        do {
            record = try await FunctionRun.run(
                function: function, inputs: URL(filePath: inputs), output: URL(filePath: output),
                repeatCount: repeatCount, source: source)
        } catch let error as FunctionError {
            throw ValidationError(error.description)
        }
        try emit(record)
        if !record.repeatIdentical { throw ExitCode.failure }
    }
}
