import Foundation

struct FunctionFailureRecord: Codable, Equatable {
    enum Kind: String, Codable {
        case load
        case call
        case write
        case repeatMismatch = "repeat_mismatch"
        case cancelled
    }

    var record = "function-failure"
    let kind: Kind

    func write(to folder: URL) {
        guard let line = try? JSONEncoder.records.encode(self) + Data("\n".utf8) else { return }
        try? line.write(to: folder.appending(path: "failure.json"), options: .atomic)
    }
}
