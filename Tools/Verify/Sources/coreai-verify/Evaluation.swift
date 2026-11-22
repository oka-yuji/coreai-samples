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
        return try decoder.decode(EvaluationRun.self, from: Data(contentsOf: runURL))
    }

    static func isUnstarted(_ folder: URL) throws -> Bool {
        let outputs = folder.appending(path: "outputs.jsonl")
        let progress = folder.appending(path: "progress.json")
        switch try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() {
        case []: return true
        case ["outputs.jsonl"]: return try isEmptyFile(outputs)
        case ["outputs.jsonl", "progress.json"]:
            return try isEmptyFile(outputs)
                && (try? JSONDecoder().decode(WrittenLines.self, from: Data(contentsOf: progress)))
                    == WrittenLines.empty
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
            (done, hasher) = try recordedLines(in: folder, inputs: items)
            current = stored
            current.items = done.count
            current.finished = nil
        } else {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try Data().write(to: outputs)
            try write(WrittenLines.empty, in: folder)
        }
        try write(current, in: folder)
        let file = try FileHandle(forWritingTo: outputs)
        defer { try? file.close() }
        try file.seekToEnd()
        for item in items.prefix(limit ?? items.count) where !done.contains(item.id) {
            let itemStarted = ContinuousClock.now
            let output = try await item.generate(with: source, stopTokens: stopTokens)
            let line = try JSONEncoder.records.encode(output) + Data("\n".utf8)
            try file.write(contentsOf: line)
            try file.synchronize()
            hasher.update(data: line)
            current.items += 1
            try write(WrittenLines(lines: current.items, sha256: hex(Data(hasher.finalize()))), in: folder)
            try emit(
                EvaluationProgress(
                    id: output.id, inputTokens: output.inputTokens, outputTokens: output.outputIds.count,
                    stop: output.error == nil ? output.stop : "error", seconds: seconds(.now - itemStarted)))
            if let maxMinutes, seconds(.now - started) >= maxMinutes * 60 { break }
        }
        current.finished = .now
        try write(current, in: folder)
        return current
    }

    static func recordedLines(in folder: URL, inputs: [EvaluationInput]) throws -> (Set<String>, SHA256) {
        let outputs = folder.appending(path: "outputs.jsonl")
        let progressURL = folder.appending(path: "progress.json")
        let data = try Data(contentsOf: outputs)
        guard FileManager.default.fileExists(atPath: progressURL.path) || data.isEmpty else {
            throw ValidationError(
                "outputs.jsonl has lines, but progress.json is missing; they were changed outside this command.")
        }
        let progress =
            data.isEmpty && !FileManager.default.fileExists(atPath: progressURL.path)
            ? WrittenLines.empty : try JSONDecoder().decode(WrittenLines.self, from: Data(contentsOf: progressURL))
        var lines: [Data] = []
        var kept = data.startIndex
        while lines.count < progress.lines, let end = data[kept...].firstIndex(of: UInt8(ascii: "\n")) {
            lines.append(data[kept..<end])
            kept = end + 1
        }
        guard lines.count == progress.lines else {
            throw ValidationError(
                "outputs.jsonl has \(lines.count) complete lines, fewer than the \(progress.lines) that progress.json records; it was changed outside this command."
            )
        }
        var hasher = SHA256()
        hasher.update(data: data[..<kept])
        guard hex(Data(hasher.finalize())) == progress.sha256 else {
            throw ValidationError(
                "The first \(progress.lines) lines of outputs.jsonl differ from progress.json; they were changed outside this command."
            )
        }
        if kept < data.endIndex {
            let file = try FileHandle(forWritingTo: outputs)
            try file.truncate(atOffset: UInt64(kept))
            try file.close()
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
        return (ids, hasher)
    }

    static func write(_ progress: WrittenLines, in folder: URL) throws {
        try JSONEncoder.records.encode(progress).write(to: folder.appending(path: "progress.json"), options: .atomic)
    }

    static func write(_ record: EvaluationRun, in folder: URL) throws {
        let encoder = JSONEncoder.records
        encoder.outputFormatting.insert(.prettyPrinted)
        try encoder.encode(record).write(to: folder.appending(path: "run.json"), options: .atomic)
    }
}
