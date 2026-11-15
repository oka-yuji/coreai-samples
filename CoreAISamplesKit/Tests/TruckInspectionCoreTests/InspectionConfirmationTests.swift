import Foundation
import Testing
import TruckInspectionCore
import TruckInspectionRuntime

struct InspectionConfirmationTests {
    struct Fixture: Decodable {
        struct Example: Decodable {
            let id: String
            let text: String
            let expected: String
            let vocabulary: InspectionVocabulary?
        }
        let vocabulary: InspectionVocabulary
        let cases: [Example]
    }

    private static func fixture() throws -> Fixture {
        let url = try #require(Bundle.module.url(forResource: "confirmation", withExtension: "json"))
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private static func exampleIDs() throws -> [String] {
        try fixture().cases.map(\.id)
    }

    @Test(
        "CONFIRM-01 and CONFIRM-02: Accept affirmative phrases without accepting contradictory replies",
        arguments: try exampleIDs())
    func confirmation(id: String) throws {
        let fixture = try Self.fixture()
        let example = try #require(fixture.cases.first { $0.id == id })
        let expectations: [String: InspectionConfirmation] = ["yes": .yes, "no": .no, "unknown": .unknown]
        let expected = try #require(expectations[example.expected])
        let vocabulary = try example.vocabulary ?? InspectionCatalogLoader.load().vocabulary
        #expect(InspectionRules.confirmation(example.text, vocabulary: vocabulary) == expected)
    }
}
