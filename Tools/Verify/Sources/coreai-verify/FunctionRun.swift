import Darwin
import Foundation

enum FunctionRun {
    static func run(
        function name: String, inputs listURL: URL, output: URL, repeatCount: Int, source: some FunctionSource,
        folderMode: mode_t = 0o755
    ) async throws -> FunctionRecord {
        precondition(repeatCount >= 1)
        let path = output.path(percentEncoded: false)
        var status = stat()
        guard lstat(path, &status) != 0 else { throw FunctionError.outputExists(path) }
        let list = try FunctionInput.read(InputFile.read(listURL, label: "the inputs file", expectedBytes: nil))
        let asset = try source.describe()
        try FunctionSignature.find(name, in: asset.functions).check(list)
        let folder = listURL.deletingLastPathComponent()
        var inputs: [FunctionTensor] = []
        for item in list {
            let data = try InputFile.read(
                folder.appending(path: item.file), label: item.file, expectedBytes: item.byteCount)
            inputs.append(FunctionTensor(name: item.name, dtype: item.dtype, shape: item.shape, data: data))
        }
        guard mkdir(path, folderMode) == 0 else {
            throw errno == EEXIST
                ? FunctionError.outputExists(path)
                : FunctionError.cannot("create the output folder", String(cString: strerror(errno)))
        }
        do {
            try FunctionFailureRecord(kind: .incomplete).write(to: output)
        } catch {
            var reason = error.localizedDescription
            if rmdir(path) != 0 {
                reason += "; the output folder could not be removed: \(String(cString: strerror(errno)))"
            }
            throw FunctionError.cannot("write the failure record", reason)
        }
        var stage = FunctionFailureRecord.Kind.load
        do {
            try Task.checkCancellation()
            let load = try await source.load(name)
            stage = .call
            var first: [FunctionTensor] = []
            var last: [FunctionTensor] = []
            var timings: [Double] = []
            var differing: [Int] = []
            for number in 1...repeatCount {
                try Task.checkCancellation()
                let start = ContinuousClock.now
                last = try await source.call(inputs)
                timings.append(seconds(ContinuousClock.now - start))
                if number == 1 { first = last } else if last != first { differing.append(number) }
            }
            stage = .write
            try Task.checkCancellation()
            let files = try last.map { try fileName(for: $0.name) }
            for (tensor, file) in zip(last, files) {
                try tensor.data.write(to: output.appending(path: file), options: .withoutOverwriting)
            }
            let record = FunctionRecord(
                mainHash: asset.mainHash, function: name, inputs: inputs.map { .init($0) },
                outputs: zip(last, files).map { .init($0, file: $1) }, load: load, callS: timings,
                differingCalls: differing)
            try (JSONEncoder.records.encode(record) + Data("\n".utf8)).write(
                to: output.appending(path: "function.json"), options: .atomic)
            if differing.isEmpty {
                try FileManager.default.removeItem(at: output.appending(path: FunctionFailureRecord.fileName))
            } else {
                try? FunctionFailureRecord(kind: .repeatMismatch).write(to: output)
            }
            return record
        } catch {
            try? FunctionFailureRecord(kind: error is CancellationError ? .cancelled : stage).write(to: output)
            throw error
        }
    }

    private static func fileName(for name: String) throws -> String {
        let file = name + ".bin"
        guard !name.isEmpty, !name.utf8.contains(0x2F), !name.utf8.contains(0), file.utf8.count <= Int(NAME_MAX) else {
            throw FunctionError.unusableOutputName(name)
        }
        return file
    }
}
