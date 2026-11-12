import Foundation

extension ChronosCatalog {
    public var isAPI: Bool { scenarios.first?.apiHistory != nil }

    public func replacingWithAPI(_ history: APIWindow, actuals: APIWindow? = nil) throws -> ChronosCatalog {
        try history.requireVerifiedHistory()
        guard history.count == 1008,
            Set(history.series.map(\.areaID)) == Set(areas.map(\.id))
        else { throw ChronosDemoFailure.invalidData }
        _ = try ForecastInputSnapshot(history: history, actuals: nil, step: 0)
        if let actuals {
            guard actuals.start == history.end, actuals.count == 16,
                actuals.verification == history.verification,
                Set(actuals.series.map(\.areaID)) == Set(areas.map(\.id)),
                actuals.datasets.allSatisfy({ dataset in
                    history.datasets.first(where: { $0.year == dataset.year }).map { $0 == dataset } ?? true
                })
            else { throw ChronosDemoFailure.invalidData }
        }
        let scene = ChronosScenario(
            id: "api-" + String(Int(history.end.timeIntervalSince1970)), name: ChronosDisplayName(ja: "API", en: "API"),
            focusArea: "230", history: [:], truth: [:], origin: history.end, apiHistory: history, apiActuals: actuals)
        return ChronosCatalog(
            areas: areas, scenarios: [scene], origin: history.end, intervalSeconds: 1800, historySteps: 1008,
            truthSteps: 16, timeZone: "America/New_York", isSynthetic: false,
            sourceURL: "https://data.cityofnewyork.us", dataPeriod: "API")
    }
}
