import CryptoKit
import Foundation

struct WrittenLines: Codable, Equatable {
    static let empty = WrittenLines(lines: 0, sha256: hex(Data(SHA256.hash(data: Data()))))

    let lines: Int
    let sha256: String
}
