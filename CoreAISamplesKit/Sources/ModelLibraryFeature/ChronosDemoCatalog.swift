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
            id: selection.demoID, title: "Taxi demand forecast preview", systemImage: "chart.xyaxis.line",
            summary:
                "Preview a planned taxi demand forecast from time-series data. Predictions are not connected in this build.",
            explanation: [
                DemoExplanation(
                    title: "How it works",
                    text:
                        "Chronos-2 predicts the next 16 values of one time series. The taxi forecast screen is not connected in this build."
                ),
                DemoExplanation(
                    title: "1. Prepare the model",
                    text:
                        "Import the complete prepared Chronos-2 FP32 folder. It uses about 479 MB, plus space for the app copy. This build does not provide a model download."
                ),
                DemoExplanation(
                    title: "Preview availability",
                    text:
                        "You can import, check and delete the model in Models. Preparing it does not start a prediction."
                ),
            ], platforms: platforms, searchTerms: searchTerms[selection.demoID])
    }
}
