import CryptoKit
import Darwin
import Foundation
import ImageGenerationCore

public actor ImageDirectoryTrace: ImageTraceSink {
    private struct Record: Encodable {
        struct Tensor: Encodable {
            let contract: ImageTensorContract
            let file: String
            let bytes: Int
            let sha256: String
            let nonfiniteElements: Int
        }
        let format = 1
        let name: String
        let byteOrder = "little-endian"
        let tensors: [Tensor]
    }

    private let directory: URL
    private var created = false
    private var recorded: Set<String> = []

    public init(directory: URL) { self.directory = directory }

    public func record(name: String, values: [String: ImageTensorData], contracts: [ImageTensorContract]) throws {
        guard safeName(name), Set(values.keys) == Set(contracts.map(\.name)), !recorded.contains(name) else {
            throw ImageExecutionError.invalidTensor("trace identity")
        }
        if !created {
            guard directory.isFileURL else { throw ImageExecutionError.invalidResource }
            guard mkdir(directory.path, 0o700) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            created = true
        }
        recorded.insert(name)
        var tensors: [Record.Tensor] = []
        for contract in contracts {
            guard safeName(contract.name), let value = values[contract.name],
                value.count == (try contract.elementCount())
            else {
                throw ImageExecutionError.invalidTensor("trace shape")
            }
            let data: Data
            let suffix: String
            let nonfinite: Int
            switch value {
            case .floats(let floats):
                guard contract.dtype == .float32 else { throw ImageExecutionError.invalidTensor("trace dtype") }
                data = floats.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }
                suffix = ".f32"
                nonfinite = floats.reduce(0) { $0 + ($1.isFinite ? 0 : 1) }
            case .integers(let integers):
                guard contract.dtype == .int32 else { throw ImageExecutionError.invalidTensor("trace dtype") }
                data = integers.map(\.littleEndian).withUnsafeBytes { Data($0) }
                suffix = ".i32"
                nonfinite = 0
            case .mask(let mask):
                guard contract.dtype == .bool else { throw ImageExecutionError.invalidTensor("trace dtype") }
                data = Data(mask)
                suffix = ".u8"
                nonfinite = 0
            }
            let file = name + "-" + contract.name + suffix
            try data.write(to: directory.appending(path: file), options: .withoutOverwriting)
            tensors.append(
                Record.Tensor(
                    contract: contract, file: file, bytes: data.count,
                    sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
                    nonfiniteElements: nonfinite))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(Record(name: name, tensors: tensors))
            .write(to: directory.appending(path: name + ".json"), options: .withoutOverwriting)
    }

    private func safeName(_ name: String) -> Bool {
        !name.isEmpty && name.utf8.count <= 100
            && name.utf8.allSatisfy { (48...57).contains($0) || (97...122).contains($0) || $0 == 45 || $0 == 95 }
    }
}
