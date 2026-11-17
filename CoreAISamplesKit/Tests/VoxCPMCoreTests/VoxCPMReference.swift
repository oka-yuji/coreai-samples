import Foundation
import Testing

struct VoxCPMReference: Decodable {
    struct Prompt: Decodable {
        let name: String
        let tokenIDs: [Int32]
        let expectedIDs: [Int32]
    }

    struct Token: Decodable {
        let id: Int32
        let token: String
    }

    struct Call: Decodable {
        let x: String
        let t: String
        let dt: String
        let velocity: String
    }

    struct Patch: Decodable {
        let patch: Int
        let ditHidden: String
        let condition: String
        let calls: [Call]
        let latent: String
    }

    let prompts: [Prompt]
    let vocabulary: [Token]
    let timeSpan: String
    let noise: [String]
    let solver: [Patch]
    let stopLogits: [String]

    static func load() throws -> VoxCPMReference {
        let url = try #require(
            Bundle.module.url(forResource: "reference", withExtension: "json", subdirectory: "Resources"))
        return try JSONDecoder().decode(VoxCPMReference.self, from: Data(contentsOf: url))
    }

    static func floats(_ base64: String) throws -> [Float] {
        let data = try #require(Data(base64Encoded: base64))
        try #require(data.count.isMultiple(of: 4))
        return data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
            }
        }
    }

    static func bits(_ values: [Float]) -> [UInt32] {
        values.map(\.bitPattern)
    }

    static func maximumNormalizedError(_ actual: [Float], _ reference: [Float]) -> Double {
        guard actual.count == reference.count, actual.allSatisfy(\.isFinite) else { return .infinity }
        let scale = max(reference.map { abs(Double($0)) }.max() ?? 0, 1e-8)
        let difference = zip(actual, reference).map { abs(Double($0) - Double($1)) }.max() ?? 0
        return difference / scale
    }

    func patch(_ number: Int) throws -> Patch {
        try #require(solver.first { $0.patch == number })
    }

    func noise(_ patch: Int) throws -> [Float] {
        try #require(noise.indices.contains(patch))
        return try Self.floats(noise[patch])
    }
}
