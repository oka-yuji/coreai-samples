import CryptoKit
import Foundation

struct OutputFileWriter {
    let function: String
    let name: String
    private let handle: FileHandle
    private var hasher = SHA256()
    private var rows = 0
    private var width = 0

    init(function: String, name: String, path: String) throws {
        self.function = function
        self.name = name
        handle = try FileHandle(forWritingTo: URL(filePath: path))
    }

    mutating func append(_ values: [Float16]) throws {
        try values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { bytes in
            hasher.update(bufferPointer: bytes)
            try handle.write(contentsOf: bytes)
        }
        rows += 1
        width = values.count
    }

    func finish() throws -> OutputFileRecord {
        try handle.close()
        return OutputFileRecord(
            function: function, name: name, rows: rows, width: width, sha256: hex(Data(hasher.finalize())))
    }
}
