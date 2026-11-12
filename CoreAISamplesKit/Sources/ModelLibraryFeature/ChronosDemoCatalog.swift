public enum ChronosDemoCatalog {
    public static let selection = DemoSelection(modelID: "chronos2", demoID: "taxi-forecast")
    public static let salesSelection = DemoSelection(modelID: "chronos2", demoID: "product-sales-forecast")

    static func model(
        artifact: ModelArtifact, platforms: Set<LibraryPlatform>, searchTerms: [String: [String]]
    ) -> LibraryModel {
        LibraryModel(
            id: selection.modelID, name: "Chronos-2", category: "Time Series Forecasting",
            searchTerms: searchTerms[selection.modelID] ?? ["time series", "forecast", "taxi"],
            artifacts: [artifact], defaultArtifactID: artifact.id, platforms: platforms,
            demos: [
                ModelDemoLink(demoID: selection.demoID, platforms: platforms),
                ModelDemoLink(demoID: salesSelection.demoID, platforms: platforms, optionalCapabilities: [.network]),
            ])
    }

    static func salesDemo(platforms: Set<LibraryPlatform>, searchTerms: [String: [String]]) -> DemoDescriptor {
        DemoDescriptor(
            id: salesSelection.demoID, title: "Product sales forecast", systemImage: "chart.bar.xaxis",
            summary: "Compare 7 or 14 days of product sales with actuals and a seasonal baseline.",
            explanation: [
                DemoExplanation(
                    title: "How it works",
                    text:
                        "Choose a product and a historical date. Predict from the preceding 365 calendar days with Chronos-2 on this device. Missing sales stay missing."
                ),
                DemoExplanation(
                    title: "Public sales data",
                    text:
                        "Load the original Mendeley Hierarchical Sales Data explicitly. The source uses CC BY-NC 3.0; UCI lists CC BY 4.0. Noncommercial source terms apply. No sales values are bundled."
                ),
                DemoExplanation(
                    title: "1. Prepare the model",
                    text:
                        "Use the same prepared Chronos-2 FP32 model as the taxi demo. Opening this demo or preparing the model never starts a download or prediction."
                ),
            ], platforms: platforms, searchTerms: searchTerms[salesSelection.demoID] ?? ["sales", "product", "SKU"],
            allowsPreviewInteraction: true)
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
                        "Download the Chronos-2 FP32 model, or import the complete prepared folder. Model files use about 479 MB. Importing requires extra space for the app copy."
                ),
                DemoExplanation(
                    title: "Fixed-data preview",
                    text:
                        "The interactive precomputed preview works without the model. Preparing the model does not start a prediction. Apple basemap tiles need network access."
                ),
            ], platforms: platforms, searchTerms: searchTerms[selection.demoID], allowsPreviewInteraction: true)
    }
}
