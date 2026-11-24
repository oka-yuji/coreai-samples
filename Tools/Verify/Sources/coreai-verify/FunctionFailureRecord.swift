import Foundation

struct FunctionFailureRecord: Codable, Equatable {
    enum Kind: String, Codable {
        case incomplete
        case load
        case call
        case write
        case repeatMismatch = "repeat_mismatch"
        case cancelled
    }

    static let fileName = "failure.json"

    var record = "function-failure"
    let kind: Kind

    func write(to folder: URL) throws {
        try (JSONEncoder.records.encode(self) + Data("\n".utf8)).write(
            to: folder.appending(path: Self.fileName), options: .atomic)
    }
}
