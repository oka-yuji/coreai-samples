import ChronosCore
import Foundation
import Testing

@testable import ForecastFeature

struct ForecastDataTests {
    @Test("S2 Synthetic quoted sales preserve the first missing date and every daily gap across DST")
    func calendar() throws {
        let data = try SalesDataset.parse(syntheticCSV())
        #expect(data.series.map(\.id) == ["B1_1", "B2_1", "B3_1", "B4_1"])
        #expect(data.timestamps.count == 1826)
        #expect(data.series.allSatisfy { $0.values[0] == nil && $0.values[1] == 1001 })
        #expect(data.series.allSatisfy { $0.values.filter { $0 != nil }.count == 3 })
        let calendar = try ForecastKind.productSales.calendar()
        #expect(
            zip(data.timestamps, data.timestamps.dropFirst()).allSatisfy {
                calendar.date(byAdding: .day, value: 1, to: $0) == $1
            })
        let intervals = zip(data.timestamps, data.timestamps.dropFirst()).map { $1.timeIntervalSince($0) }
        #expect(intervals.contains(23 * 3600) && intervals.contains(25 * 3600))
    }

    @Test("S2 Strict parsing rejects malformed columns, dates, numbers and promotion flags")
    func invalidSource() throws {
        let valid = String(decoding: syntheticCSV(), as: UTF8.self)
        let cases = [
            valid.replacingOccurrences(of: "QTY_B1_2", with: "QTY_B1_1"),
            valid.replacingOccurrences(of: "QTY_B1_42", with: "QTY_B1_43"),
            valid.replacingOccurrences(of: "2014-01-02", with: "2014-02-30"),
            valid.replacingOccurrences(of: "2015-03-29", with: "2014-01-02"),
            valid.replacingOccurrences(of: "2015-03-29", with: "2014-01-01"),
            valid.replacingOccurrences(of: "2014-01-02", with: "2014-1-02"),
            valid.replacingOccurrences(of: "2014-01-02", with: "2014-01-02junk"),
            valid.replacingOccurrences(of: "1001", with: "-1"),
            valid.replacingOccurrences(of: "1001", with: "nan"),
            valid.replacingOccurrences(of: "1001", with: "inf"),
            valid.replacingOccurrences(of: "1001", with: "1e400"),
            valid.replacingOccurrences(of: "1001", with: String(repeating: "9", count: 45)),
            valid.replacingOccurrences(of: "1001", with: "1.5"),
            valid.replacingOccurrences(of: "1001", with: ""),
            valid.replacingOccurrences(of: "\"0\"", with: "\"2\""),
            valid.replacingOccurrences(of: "\"1001\"", with: "\"1001\"junk"),
            valid.replacingOccurrences(of: "\"1001\"", with: "10\"01"),
            valid + "\"unterminated", valid + "\n", valid.replacingOccurrences(of: ",\"PROMO_B4_10\"", with: ""),
        ]
        for value in cases {
            #expect(throws: (any Error).self) { try SalesDataset.parse(Data(value.utf8)) }
        }
        #expect(throws: (any Error).self) { try SalesDataset.parse(Data([0xFF])) }
        #expect(try ForecastCSV.parse(Data("\"a,b\",\"a\"\"b\",\"a\nb\"\r\n".utf8)) == [["a,b", "a\"b", "a\nb"]])
    }

    @Test("S2 Fourteen day seasonal forecasts repeat only the last observed week without future leakage")
    func windows() throws {
        let data = try syntheticDataset()
        let request = try data.request(seriesID: "B1_1", originIndex: 400, horizon: 14)
        #expect(request.context == Array(data.series[0].values[35..<400]))
        #expect(request.actuals == Array(data.series[0].values[400..<414]))
        #expect(request.baseline == Array(data.series[0].values[393..<400]) + Array(data.series[0].values[393..<400]))
        #expect(request.baseline[0] == nil && request.baseline[7] == nil)
        var changed = data.series[0].values
        for index in 400..<changed.count { changed[index] = 9000 }
        let future = try ForecastDataset(
            kind: data.kind, timestamps: data.timestamps, series: [ForecastSeries(id: "B1_1", values: changed)])
        let another = try future.request(seriesID: "B1_1", originIndex: 400, horizon: 14)
        #expect(request.context == another.context && request.baseline == another.baseline)
        #expect(request.actuals != another.actuals)
        for (origin, horizon) in [(-1, 7), (364, 14), (499, 7), (400, 28), (Int.max, 7)] {
            #expect(throws: ForecastFailure.invalidRequest) {
                try data.request(seriesID: "B1_1", originIndex: origin, horizon: horizon)
            }
        }
        #expect(throws: ForecastFailure.invalidRequest) {
            try data.request(seriesID: "unknown", originIndex: 400, horizon: 7)
        }
        #expect(try data.request(seriesID: "B1_1", originIndex: 486, horizon: 14).actuals.count == 14)
    }

    @Test(
        "S2 Dataset admission rejects inconsistent lengths, nonfinite values and unordered timestamps before indexing")
    func validation() throws {
        let data = try syntheticDataset()
        for values: [Float?] in [[1], Array(repeating: .infinity, count: 500), Array(repeating: -1, count: 500)] {
            #expect(throws: (any Error).self) {
                try ForecastDataset(
                    kind: .productSales, timestamps: data.timestamps, series: [.init(id: "X", values: values)])
            }
        }
        var dates = data.timestamps
        dates[2] = dates[1]
        #expect(throws: (any Error).self) {
            try ForecastDataset(kind: .productSales, timestamps: dates, series: data.series)
        }
        #expect(throws: (any Error).self) {
            try ForecastDataset(kind: .productSales, timestamps: data.timestamps, series: data.series + data.series)
        }
    }

    @Test("S2 Metrics compare identical valid triples and leave zero denominator WAPE unavailable")
    func metrics() {
        let result = ForecastMetrics(
            actuals: [2, 0, nil, 10, 3], model: [-3, 1, 4, nil, 5], baseline: [1, 0, 2, 3, nil])
        #expect(result.count == 2 && result.modelMAE == 1.5 && result.baselineMAE == 0.5)
        #expect(result.modelWAPE == 1.5 && result.baselineWAPE == 0.5)
        let zero = ForecastMetrics(actuals: [0], model: [-2], baseline: [1])
        #expect(zero.count == 1 && zero.modelMAE == 0 && zero.baselineMAE == 1)
        #expect(zero.modelWAPE == nil && zero.baselineWAPE == nil)
        let missing = ForecastMetrics(actuals: [nil, .nan], model: [1, 2], baseline: [1, 2])
        #expect(missing.count == 0 && missing.modelMAE == nil)
        #expect(ForecastMetrics(actuals: [1], model: [], baseline: [1]).count == 0)
    }

    @Test("S2 Presentation clamps only the requested quantiles and retains every raw output and identity")
    func output() throws {
        let data = try syntheticDataset()
        let request = try data.request(seriesID: "B1_1", originIndex: 400, horizon: 7)
        let raw = try ChronosPrediction(quantiles: (0..<21).map { Array(repeating: Float($0 - 11), count: 16) })
        let result = try ForecastResult(
            request: request, raw: raw, identity: syntheticIdentity(), seconds: 1, loaded: true,
            timingBoundary: "Synthetic")
        #expect(result.median == Array(repeating: 0, count: 7))
        #expect(result.lower == Array(repeating: 0, count: 7))
        #expect(result.upper == Array(repeating: 7, count: 7))
        #expect(result.raw == raw && raw.quantiles[10][0] == -1 && result.identity == syntheticIdentity())
    }
}

func syntheticCSV() -> Data {
    let items = [42, 45, 21, 10].enumerated().flatMap { brand, count in
        (1...count).map { "B\(brand + 1)_\($0)" }
    }
    let headers = ["DATE"] + items.map { "QTY_" + $0 } + items.map { "PROMO_" + $0 }
    let rows =
        [headers]
        + ["2014-01-02", "2015-03-29", "2018-12-31"].map { date in
            [date] + Array(repeating: "1001", count: 118) + Array(repeating: "0", count: 118)
        }
    return Data(
        (rows.map { $0.map { "\"" + $0 + "\"" }.joined(separator: ",") }.joined(separator: "\r\n") + "\r\n").utf8)
}

func syntheticDataset() throws -> ForecastDataset {
    let calendar = try ForecastKind.productSales.calendar()
    let first = try #require(calendar.date(from: DateComponents(year: 2014, month: 1, day: 1)))
    let dates = try (0..<500).map { try #require(calendar.date(byAdding: .day, value: $0, to: first)) }
    var values: [Float?] = (0..<500).map { Float(1000 + $0) }
    values[393] = nil
    return try ForecastDataset(
        kind: .productSales, timestamps: dates,
        series: [ForecastSeries(id: "B1_1", values: values), ForecastSeries(id: "B2_1", values: values)])
}

func syntheticIdentity() -> ChronosBundleIdentity {
    ChronosBundleIdentity(
        source: .expected, manifest: .init(bytes: 1, sha256: String(repeating: "a", count: 64)), files: [:])
}
