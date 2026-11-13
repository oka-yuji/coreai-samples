import ForecastFeature
import Foundation
import Testing

@testable import ForecastRuntime

struct BikeRepositoryTests {
    @Test("B1 Bundled source preserves all hours, weather units and the 295 observed shutdown zeros")
    func bundled() async throws {
        let repository = BikeDataRepository()
        let dataset = try await repository.load()
        #expect(dataset.kind == .seoulBike && dataset.timestamps.count == 8760)
        #expect(dataset.series.map(\.id) == [BikeDataset.seriesID])
        #expect(dataset.series[0].values.first == 254 && dataset.series[0].values.last == 584)
        #expect(zip(dataset.timestamps, dataset.timestamps.dropFirst()).allSatisfy { $1.timeIntervalSince($0) == 3600 })
        let calendar = try ForecastKind.seoulBike.calendar()
        #expect(
            calendar.dateComponents([.year, .month, .day, .hour], from: dataset.timestamps[0])
                == DateComponents(year: 2017, month: 12, day: 1, hour: 0))
        #expect(
            calendar.dateComponents([.year, .month, .day, .hour], from: dataset.timestamps[8759])
                == DateComponents(year: 2018, month: 11, day: 30, hour: 23))
        var shutdowns: Set<Date> = []
        for origin in Array(stride(from: 336, through: 8736, by: 336)) + [8759] {
            let weather = try dataset.pastWeather(originIndex: origin, count: 336)
            #expect(weather.map(\.timestamp) == Array(dataset.timestamps[(origin - 336)..<origin]))
            for (index, item) in weather.enumerated() where !item.isFunctioning {
                shutdowns.insert(item.timestamp)
                #expect(dataset.series[0].values[origin - 336 + index] == 0)
            }
        }
        #expect(shutdowns.count == 295)
        let first = try #require(dataset.pastWeather(originIndex: 336, count: 336).first)
        #expect(first.temperature == -5.2 && first.humidity == 37 && first.windSpeed == 2.2)
        #expect(first.visibility == 2000 && first.dewPoint == -17.6 && first.solarRadiation == 0)
        #expect(first.rainfall == 0 && first.snowfall == 0 && first.season == "Winter")
        #expect(first.isFunctioning && !first.isHoliday)
        #expect(try await repository.load(refresh: true) == dataset)
        for (origin, count) in [(335, 1), (8760, 1), (Int.max, 1), (1000, 0), (1000, 337), (1000, Int.max)] {
            #expect(throws: ForecastFailure.invalidRequest) {
                try dataset.pastWeather(originIndex: origin, count: count)
            }
        }
    }

    @Test("B1 Invalid source dates, hours, counts, weather, labels and CSV structure fail admission")
    func invalidSource() throws {
        let original = try BikeSource.read(#require(BikeSource.bundledDirectory))
        let text = String(decoding: original, as: UTF8.self)
        let rows = text.components(separatedBy: "\r\n")
        let changes: [(Int, String)] = [
            (0, "31/02/2017"), (0, "01/12/9999"), (0, "1/12/2017"), (0, "01/13/2017"),
            (1, "-1"), (1, "1.5"), (1, "NaN"), (1, String(repeating: "9", count: 40)),
            (2, "24"), (2, "-1"), (2, "1.5"), (2, "1"), (2, " 0"),
            (3, "nan"), (4, "inf"), (5, "-inf"), (6, ""), (7, "1e40"), (8, "nan"), (9, "nan"), (10, "inf"),
            (11, "Unknown"), (12, "Unknown"), (13, "Unknown"), (13, "No"),
        ]
        for (column, value) in changes {
            var altered = rows
            var fields = altered[1].components(separatedBy: ",")
            fields[column] = value
            altered[1] = fields.joined(separator: ",")
            #expect(throws: ForecastFailure.invalidData) {
                try BikeDataset.parse(Data(altered.joined(separator: "\r\n").utf8))
            }
        }
        for bad in [
            text.replacingOccurrences(of: "Temperature(°C)", with: "Temperature(F)"),
            rows.dropLast(2).joined(separator: "\r\n"), text + rows[1] + "\r\n",
            text.replacingOccurrences(of: "\r\n", with: "\r"), text + "\"unfinished",
        ] {
            #expect(throws: ForecastFailure.invalidData) { try BikeDataset.parse(Data(bad.utf8)) }
        }
        #expect(throws: ForecastFailure.invalidData) { try BikeDataset.parse(Data([0xFF])) }
    }

    @Test("B2 Future weather and rental changes cannot alter the input, weekly baseline or past weather")
    func futureIsolation() throws {
        let source = try BikeSource.read(#require(BikeSource.bundledDirectory))
        let dataset = try BikeDataset.parse(source)
        let origin = 1000
        let before = try dataset.request(seriesID: BikeDataset.seriesID, originIndex: origin, horizon: 16)
        var rows = String(decoding: source, as: UTF8.self).components(separatedBy: "\r\n")
        for index in origin..<8760 {
            var fields = rows[index + 1].components(separatedBy: ",")
            fields[3] = "50"
            fields[9] = "100"
            fields[12] = "Holiday"
            fields[13] = "No"
            fields[1] = "0"
            rows[index + 1] = fields.joined(separator: ",")
        }
        let changed = try BikeDataset.parse(Data(rows.joined(separator: "\r\n").utf8))
        let after = try changed.request(seriesID: BikeDataset.seriesID, originIndex: origin, horizon: 16)
        #expect(before.context == after.context && before.baseline == after.baseline)
        #expect(before.actuals != after.actuals && after.actuals.allSatisfy { $0 == 0 })
        #expect(
            try dataset.pastWeather(originIndex: origin, count: 336)
                == changed.pastWeather(originIndex: origin, count: 336))
    }

    @Test("B1 A missing, changed, oversized or linked resource is rejected without replacing source bytes")
    func sourceIdentity() async throws {
        let root = try ForecastFixture.directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bundled = try #require(BikeSource.bundledDirectory)
        for name in ["SeoulBikeData.csv", "manifest.json", "NOTICE.txt"] {
            let directory = root.appending(path: name)
            try FileManager.default.copyItem(at: bundled, to: directory)
            let file = directory.appending(path: name)
            var data = try Data(contentsOf: file)
            data[0] ^= 1
            try data.write(to: file)
            let repository = BikeDataRepository(directory: directory)
            await #expect(throws: ForecastFailure.sourceMismatch) { try await repository.load() }
            try Data(repeating: 65, count: 700_000).write(to: file)
            await #expect(throws: ForecastFailure.sourceMismatch) { try await repository.load() }
            try FileManager.default.removeItem(at: file)
            await #expect(throws: ForecastFailure.sourceMismatch) { try await repository.load() }
            try FileManager.default.createSymbolicLink(at: file, withDestinationURL: bundled.appending(path: name))
            await #expect(throws: ForecastFailure.sourceMismatch) { try await repository.load() }
        }
        #expect(try BikeSource.read(bundled).count == BikeSource.normalizedBytes)
    }
}
