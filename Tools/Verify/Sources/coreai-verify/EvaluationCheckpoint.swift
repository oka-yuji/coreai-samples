import CryptoKit
import Foundation

struct EvaluationCheckpoint: Codable, Equatable {
    static let emptyDigest = hex(Data(SHA256.hash(data: Data())))

    let lines: Int
    let sha256: String
    var nextSha256: String?
    let run: String
}
