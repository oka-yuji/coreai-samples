import Foundation
import Testing
import TruckInspectionCore
import TruckInspectionRuntime

struct InspectionRulesTests {
    struct Examples: Decodable {
        struct Answer: Decodable {
            let item: String
            let text: String
            let status: InspectionStatus
        }
        struct Confirmation: Decodable {
            let text: String
            let expected: String
        }
        let answers: [Answer]
        let unmatched: [String]
        let confirmation: [Confirmation]
        let partial: String
        let final: String
    }

    private func examples() throws -> Examples {
        let url = try #require(Bundle.module.url(forResource: "answers", withExtension: "json"))
        return try JSONDecoder().decode(Examples.self, from: Data(contentsOf: url))
    }

    @Test("TC1: Load the appropriate questions for hydraulic and air brake trucks")
    func vehicleQuestions() throws {
        let catalog = try InspectionCatalogLoader.load()
        let hydraulic = Set(catalog.items(for: .hydraulic).map(\.id))
        let air = Set(catalog.items(for: .airBrake).map(\.id))
        #expect(catalog.items.count == 23)
        #expect(hydraulic.count == 18)
        #expect(air.count == 21)
        #expect(hydraulic.contains("brake_fluid"))
        #expect(!air.contains("brake_fluid"))
        #expect(!hydraulic.contains("air_pressure"))
        #expect(air.contains("air_pressure"))
        #expect(catalog.items.filter(\.conditionBased).count == 11)
        for item in catalog.items {
            #expect(catalog.prompt(for: item, retry: true).contains(item.title))
            #expect(catalog.confirmation(for: item, status: .attention).contains(item.title))
            #expect(!catalog.hints(for: item, confirming: false).isEmpty)
        }
    }

    @Test("TC2: Resolve exact choices and item aliases without accepting ambiguous confirmation")
    func aliasesAndConfirmation() throws {
        let catalog = try InspectionCatalogLoader.load()
        let examples = try examples()
        for answer in examples.answers {
            let item = try #require(catalog.items.first { $0.id == answer.item })
            let result = try InspectionRules.exactAnswer(answer.text, item: item, catalog: catalog)
            let candidate = try #require(result)
            #expect(candidate.status == answer.status)
            #expect(candidate.method == .exact)
        }
        let item = try #require(catalog.items.first { $0.id == "tire_pressure" })
        for text in examples.unmatched {
            #expect(try InspectionRules.exactAnswer(text, item: item, catalog: catalog) == nil)
        }
        let oilOnly = try #require(examples.answers.first { $0.item == "engine_oil" })
        #expect(try InspectionRules.exactAnswer(oilOnly.text, item: item, catalog: catalog) == nil)
        for example in examples.confirmation {
            let expected: InspectionConfirmation =
                switch example.expected {
                case "yes": .yes
                case "no": .no
                default: .unknown
                }
            #expect(InspectionRules.confirmation(example.text, vocabulary: catalog.vocabulary) == expected)
        }
    }

    @Test("TC3: Preserve the last nonempty partial when the final is empty")
    func emptyFinal() throws {
        let examples = try examples()
        for final: String? in [nil, "", " \n "] {
            #expect(InspectionRules.resolveFinal(final, lastPartial: examples.partial) == examples.partial)
        }
        #expect(InspectionRules.resolveFinal(examples.final, lastPartial: examples.partial) == examples.final)
        #expect(InspectionRules.resolveFinal(nil, lastPartial: " \n").isEmpty)
        let catalog = try InspectionCatalogLoader.load()
        let item = try #require(catalog.items.first)
        #expect(throws: InspectionError.emptyAnswer) {
            try InspectionRules.exactAnswer(" \n", item: item, catalog: catalog)
        }
        #expect(throws: InspectionError.answerTooLong) {
            try InspectionRules.exactAnswer(String(repeating: "a", count: 241), item: item, catalog: catalog)
        }
    }

    @Test("TC4: Reject malformed classification and confirm every model candidate")
    func classification() throws {
        let catalog = try InspectionCatalogLoader.load()
        for scores: [Float] in [
            [], [0.9], [.nan, 0, 0, 0], [1.1, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0],
        ] {
            #expect(throws: InspectionError.invalidClassification) {
                try InspectionRules.classify(scores, choices: catalog.choices)
            }
        }
        for index in catalog.choices.indices {
            var scores = Array(repeating: Float(0.1), count: catalog.choices.count)
            scores[index] = 1
            let candidate = try InspectionRules.classify(scores, choices: catalog.choices)
            #expect(candidate.status == catalog.choices[index].status)
            #expect(candidate.needsConfirmation)
        }
    }

    @Test("Listening waits for first speech, ends after silence and bounds the final wait")
    func listeningDeadlines() {
        var window = InspectionListeningWindow()
        #expect(window.tick(elapsed: 1.3) == .keepListening)
        window.partial("answer", elapsed: 2)
        window.partial("", elapsed: 2.5)
        window.partial("answer", elapsed: 3)
        #expect(window.lastPartial == "answer")
        #expect(window.tick(elapsed: 3.3) == .endAudio)
        #expect(window.tick(elapsed: 4) == .keepListening)
        #expect(window.tick(elapsed: 5.4) == .resolve)
        var silent = InspectionListeningWindow()
        #expect(silent.tick(elapsed: 6) == .endAudio)
        #expect(silent.tick(elapsed: 8) == .resolve)
        #expect(silent.lastPartial.isEmpty)
        #expect(InspectionRules.speechDeadline(for: "short") == 8)
        #expect(InspectionRules.speechDeadline(for: String(repeating: "a", count: 100)) == 38)
    }

    @Test("Reject incompatible catalogs and conflicting aliases")
    func invalidCatalog() throws {
        let catalog = try InspectionCatalogLoader.load()
        let data = try JSONEncoder().encode(catalog)
        var value = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        value["version"] = 2
        #expect(throws: InspectionError.invalidCatalog) {
            try InspectionCatalog.decode(JSONSerialization.data(withJSONObject: value))
        }
        value["version"] = 1
        var items = try #require(value["items"] as? [[String: Any]])
        items.append(try #require(items.first))
        value["items"] = items
        #expect(throws: InspectionError.invalidCatalog) {
            try InspectionCatalog.decode(JSONSerialization.data(withJSONObject: value))
        }
        items.removeLast()
        items[0]["aliases"] = ["attention": [try #require(catalog.choices.first).title]]
        value["items"] = items
        #expect(throws: InspectionError.invalidCatalog) {
            try InspectionCatalog.decode(JSONSerialization.data(withJSONObject: value))
        }
    }
}
