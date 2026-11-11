public enum ChronosDemoCatalog {
    public static let selection = DemoSelection(modelID: "chronos2", demoID: "taxi-forecast")

    static func model(
        artifact: ModelArtifact, platforms: Set<LibraryPlatform>, searchTerms: [String: [String]]
    ) -> LibraryModel {
        LibraryModel(
            id: selection.modelID, name: "Chronos-2", category: "Time Series Forecasting",
            searchTerms: searchTerms[selection.modelID] ?? ["time series", "forecast", "taxi"],
            artifacts: [artifact], defaultArtifactID: artifact.id, platforms: platforms,
            demos: [ModelDemoLink(demoID: selection.demoID, platforms: platforms)])
    }

    static func demo(platforms: Set<LibraryPlatform>, searchTerms: [String: [String]]) -> DemoDescriptor {
        DemoDescriptor(
            id: selection.demoID, title: "NYC taxi pickup forecast", systemImage: "chart.xyaxis.line",
            summary:
                "Explore fixed scenes with a precomputed preview, or select verified NYC API dates for an on-device forecast.",
            explanation: [
                DemoExplanation(
                    title: "How it works",
                    text:
                        "Explore 20 Manhattan taxi zones in three fixed scenes or independently verified January 2025 API data. Forecast origins run from January 22 through February 1 at 00:00 New York time. Fetch 21 days of history, predict on this device, then fetch actuals to compare."
                ),
                DemoExplanation(
                    title: "1. Prepare the model",
                    text:
                        "Import the complete prepared Chronos-2 FP32 folder. It uses about 479 MB, plus space for the app copy. This build does not provide a model download."
                ),
                DemoExplanation(
                    title: "Fixed-data preview",
                    text:
                        "The interactive precomputed preview works without the model. Preparing the model does not start a prediction. Apple basemap tiles need network access."
                ),
            ], platforms: platforms, searchTerms: searchTerms[selection.demoID], allowsPreviewInteraction: true)
    }
}
