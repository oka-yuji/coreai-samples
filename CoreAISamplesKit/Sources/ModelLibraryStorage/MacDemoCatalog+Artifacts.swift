import ModelLibraryFeature

extension MacDemoCatalog {
    public static func make(searchTerms: [String: [String]] = [:]) throws -> ModelCatalog {
        try make(
            searchTerms: searchTerms, speechArtifact: IrodoriModelArtifact.makeReferenceDownload(platforms: [.macOS]),
            ocrArtifact: TeleOCRModelArtifact.make(),
            qwenArtifact: LanguageModelArtifacts.qwen(), clefArtifact: LanguageModelArtifacts.clef(),
            jevArtifact: JEVModelArtifact.make(), chronosArtifact: ChronosModelArtifact.make())
    }
}
