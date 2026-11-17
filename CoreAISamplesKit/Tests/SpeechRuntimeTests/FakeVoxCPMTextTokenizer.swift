import VoxCPMCore

@testable import SpeechRuntime

final class FakeVoxCPMTextTokenizer: VoxCPMTextTokenizer {
    static let offset: Int32 = 1000
    let vocabulary = Dictionary(
        uniqueKeysWithValues: (32...126).map {
            (FakeVoxCPMTextTokenizer.offset + Int32($0), String(UnicodeScalar(UInt8($0))))
        })
    let log: LifetimeLog
    init(log: LifetimeLog) { self.log = log }
    func tokenIDs(for text: String) throws -> [Int32] {
        log.add("tokenize:" + text)
        return try text.unicodeScalars.map { scalar in
            let id = Self.offset + Int32(scalar.value)
            guard vocabulary[id] != nil else { throw VoxCPMError.unknownToken }
            return id
        }
    }
}
