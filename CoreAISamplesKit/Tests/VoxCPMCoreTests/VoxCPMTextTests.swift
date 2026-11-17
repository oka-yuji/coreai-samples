import Foundation
import Testing

@testable import VoxCPMCore

struct VoxCPMTextTests {
    struct Fixture: Decodable {
        struct Case: Decodable {
            let name: String
            let text: String
            let normalized: String?
        }

        let format: Int
        let python: String
        let unicode: String
        let isspaceMatches: Bool
        let whitespace: [UInt32]
        let cases: [Case]
    }

    static func fixture() throws -> Fixture {
        let url = try #require(
            Bundle.module.url(forResource: "text-normalization", withExtension: "json", subdirectory: "Resources"))
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    @Test("The whitespace set equals the code points that Python re matches with \\s in the fixture")
    func whitespaceSet() throws {
        let fixture = try Self.fixture()
        #expect(fixture.format == 1 && fixture.isspaceMatches && fixture.whitespace.count == 29)
        #expect(VoxCPMText.whitespace.map(\.value).sorted() == fixture.whitespace)
    }

    @Test("Each fixture text is normalized like the original implementation and blank text is rejected")
    func normalization() throws {
        let fixture = try Self.fixture()
        #expect(fixture.cases.count > 60)
        for item in fixture.cases {
            if let expected = item.normalized {
                let actual = try VoxCPMText.normalized(item.text)
                #expect(Array(actual.unicodeScalars) == Array(expected.unicodeScalars), "\(item.name)")
            } else {
                #expect(throws: VoxCPMError.emptyText, "\(item.name)") { try VoxCPMText.normalized(item.text) }
            }
        }
    }
}
