import Foundation
import Testing

@testable import ForecastUI

@MainActor
struct ForecastLocalizationTests {
    @Test(
        "B6 Forecast labels follow the selected language across repeated switches",
        arguments: [
            "History", "Actuals", "Median", "Seasonal naive", "Missing", "Unavailable", "Temperature (°C)",
            "Rainfall (mm)",
        ])
    func language(_ key: String) {
        let english = Locale(identifier: "en")
        let japanese = Locale(identifier: "ja")
        #expect(ForecastCopy.text(key, locale: english) == key)
        let translated = ForecastCopy.text(key, locale: japanese)
        #expect(!translated.isEmpty && translated != key)
        #expect(ForecastCopy.text(key, locale: english) == key)
    }

    @Test("B6 Unavailable forecast metrics use the selected language")
    func unavailable() {
        let english = Locale(identifier: "en")
        let japanese = Locale(identifier: "ja")
        #expect(ForecastCopy.number(nil, locale: english) == "Unavailable")
        #expect(ForecastCopy.percent(nil, locale: english) == "Unavailable")
        #expect(ForecastCopy.number(nil, locale: japanese) != "Unavailable")
        #expect(ForecastCopy.percent(nil, locale: japanese) != "Unavailable")
    }
}
