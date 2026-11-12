import ChronosFeature
import Foundation

enum ChronosJournalCodec {
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .deferredToDate
        return try encoder.encode(value)
    }

    static func decode<T: Decodable>(_ type: T.Type, _ data: Data) throws -> T {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .deferredToDate
        do { return try decoder.decode(type, from: data) } catch { throw ChronosDemoFailure.invalidData }
    }
}
