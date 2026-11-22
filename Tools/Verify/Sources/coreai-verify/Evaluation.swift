import ArgumentParser
import CryptoKit
import Foundation

enum Evaluation {
    private struct WrittenID: Decodable {
        let id: String
    }

    static func stored(in folder: URL) throws -> EvaluationRun? {
        var isFolder: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &isFolder) else {
            let parent = folder.deletingLastPathComponent()
            guard FileManager.default.fileExists(atPath: parent.path, isDirectory: &isFolder), isFolder.boolValue else {
                throw ValidationError("The folder that holds --output must exist: \(parent.path)")
            }
            return nil
        }
        let runURL = folder.appending(path: "run.json")
        guard isFolder.boolValue, FileManager.default.fileExists(atPath: runURL.path) else {
            guard isFolder.boolValue, try isUnstarted(folder) else {
                throw ValidationError("--output exists but is not a run folder with run.json: \(folder.path)")
            }
            return nil
        }
        let decoder = JSONDecoder.snake
        decoder.dateDecodingStrategy = .iso8601
        do {
            return try decoder.decode(EvaluationRun.self, from: Data(contentsOf: runURL))
        } catch let error as DecodingError {
            throw ValidationError(
                "run.json could not be read as a run record of this command, and a run folder of an earlier version cannot be resumed: \(error)"
            )
        }
    }

    static func isUnstarted(_ folder: URL) throws -> Bool {
        let outputs = folder.appending(path: "outputs.jsonl")
        let progress = folder.appending(path: "progress.json")
        switch try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() {
        case []: return true
        case ["outputs.jsonl"]: return try isEmptyFile(outputs)
        case ["outputs.jsonl", "progress.json"]:
            guard try isEmptyFile(outputs),
                let checkpoint = try? JSONDecoder.snake.decode(
                    EvaluationCheckpoint.self, from: Data(contentsOf: progress))
            else { return false }
            return checkpoint.lines == 0 && checkpoint.sha256 == EvaluationCheckpoint.emptyDigest
                && checkpoint.nextSha256 == nil
        default: return false
        }
    }

    static func isEmptyFile(_ url: URL) throws -> Bool {
        let file = try FileManager.default.attributesOfItem(atPath: url.path)
        return file[.type] as? FileAttributeType == .typeRegular && (file[.size] as? NSNumber)?.intValue == 0
    }

    static func run(
        _ items: [EvaluationInput], limit: Int? = nil, in folder: URL, record: EvaluationRun,
        source: some LogitsSource, stopTokens: Set<Int32>, started: ContinuousClock.Instant = .now,
        maxMinutes: Double? = nil
    ) async throws -> EvaluationRun {
        let outputs = folder.appending(path: "outputs.jsonl")
        var current = record
        var done: Set<String> = []
        var hasher = SHA256()
        if let stored = try stored(in: folder) {
            let differing = record.differences(from: stored)
            guard differing.isEmpty else {
                throw ValidationError(
                    "--output holds another run; these differ from its run.json: \(differing.joined(separator: ", ")).")
            }
            (done, hasher) = try recordedLines(in: folder, inputs: items, run: stored)
            current = stored
            current.items = done.count
            current.finished = nil
        } else {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try Data().write(to: outputs)
            try write(EvaluationCheckpoint(lines: 0, sha256: digest(hasher), run: record.identityDigest()), in: folder)
        }
        let identity = try current.identityDigest()
        try write(current, in: folder)
        let file = try FileHandle(forWritingTo: outputs)
        defer { try? file.close() }
        try file.seekToEnd()
        for item in items.prefix(limit ?? items.count) where !done.contains(item.id) {
            let itemStarted = ContinuousClock.now
            let output = try await item.generate(with: source, stopTokens: stopTokens)
            let line = try JSONEncoder.records.encode(output) + Data("\n".utf8)
            var next = hasher
            next.update(data: line)
            try write(
                EvaluationCheckpoint(
                    lines: current.items, sha256: digest(hasher), nextSha256: digest(next), run: identity),
                in: folder)
            try file.write(contentsOf: line)
            try file.synchronize()
            hasher = next
            current.items += 1
            try emit(
                EvaluationProgress(
                    id: output.id, inputTokens: output.inputTokens, outputTokens: output.outputIds.count,
                    stop: output.error == nil ? output.stop : "error", seconds: seconds(.now - itemStarted)))
            if let maxMinutes, seconds(.now - started) >= maxMinutes * 60 { break }
        }
        try write(EvaluationCheckpoint(lines: current.items, sha256: digest(hasher), run: identity), in: folder)
        current.finished = .now
        try write(current, in: folder)
        return current
    }

    static func recordedLines(in folder: URL, inputs: [EvaluationInput], run stored: EvaluationRun) throws -> (
        Set<String>, SHA256
    ) {
        let outputs = folder.appending(path: "outputs.jsonl")
        let progress = folder.appending(path: "progress.json")
        guard FileManager.default.fileExists(atPath: progress.path) else {
            throw ValidationError("progress.json is missing; the run folder was changed outside this command.")
        }
        let checkpoint = try JSONDecoder.snake.decode(EvaluationCheckpoint.self, from: Data(contentsOf: progress))
        guard checkpoint.run == (try stored.identityDigest()) else {
            throw ValidationError(
                "progress.json belongs to another run; the run folder was changed outside this command.")
        }
        guard (0...inputs.count).contains(stored.items) else {
            throw ValidationError(
                "run.json records \(stored.items) finished items, not a count from 0 to \(inputs.count); the run folder was changed outside this command."
            )
        }
        let newline = UInt8(ascii: "\n")
        let data = try Data(contentsOf: outputs)
        let complete = data.lastIndex(of: newline).map { $0 + 1 } ?? data.startIndex
        let lines = data[..<complete].split(separator: newline, omittingEmptySubsequences: false).dropLast()
        guard lines.count >= stored.items else {
            throw ValidationError(
                "outputs.jsonl has \(lines.count) complete lines, fewer than the \(stored.items) finished items that run.json records; the run folder was changed outside this command."
            )
        }
        var hasher = SHA256()
        hasher.update(data: data[..<complete])
        let written = digest(hasher)
        let counted = (0...lines.count).contains(checkpoint.lines)
        let confirmed = counted && lines.count == checkpoint.lines && written == checkpoint.sha256
        let announced = counted && lines.count - 1 == checkpoint.lines && written == checkpoint.nextSha256
        guard confirmed || announced else {
            throw ValidationError(
                "outputs.jsonl has \(lines.count) complete lines, which match neither the \(checkpoint.lines) lines that progress.json records nor the line that it announced; the run folder was changed outside this command."
            )
        }
        let known = Set(inputs.map(\.id))
        var ids: Set<String> = []
        for (index, line) in lines.enumerated() {
            let id = try? JSONDecoder().decode(WrittenID.self, from: line).id
            let place = "outputs.jsonl line \(index + 1)" + (id.map { " (id \($0))" } ?? "")
            guard let id, known.contains(id) else {
                throw ValidationError("\(place) has no id of the input records.")
            }
            guard ids.insert(id).inserted else { throw ValidationError("\(place) repeats an id.") }
        }
        if complete < data.endIndex {
            let file = try FileHandle(forWritingTo: outputs)
            try file.truncate(atOffset: UInt64(complete))
            try file.close()
        }
        return (ids, hasher)
    }

    static func digest(_ hasher: SHA256) -> String { hex(Data(hasher.finalize())) }

    static func write(_ checkpoint: EvaluationCheckpoint, in folder: URL) throws {
        try JSONEncoder.records.encode(checkpoint).write(to: folder.appending(path: "progress.json"), options: .atomic)
    }

    static func write(_ record: EvaluationRun, in folder: URL) throws {
        let encoder = JSONEncoder.records
        encoder.outputFormatting.insert(.prettyPrinted)
        try encoder.encode(record).write(to: folder.appending(path: "run.json"), options: .atomic)
    }
}
