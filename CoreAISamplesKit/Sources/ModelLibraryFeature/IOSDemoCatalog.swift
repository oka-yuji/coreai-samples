public enum IOSDemoCatalog {
    public static func make(
        searchTerms: [String: [String]] = [:], localImageArtifacts: [ModelArtifact]? = nil,
        speechArtifact: ModelArtifact? = nil, referenceSpeechArtifact: ModelArtifact? = nil,
        chronosArtifact: ModelArtifact? = nil
    ) throws -> ModelCatalog {
        let images = localImageArtifacts ?? verifiedImageArtifacts
        for artifact in images {
            guard artifact.sourceKind == .localManifest, artifact.isSourceValid,
                artifact.contract == "qwen-image21-mobile-v2",
                artifact.upstream?.repository == "Qwen/Qwen-Image-2.1-Turbo",
                artifact.upstream?.revision == "fc13ba1986fcb905854dc8de2f52eee395f49b5b"
            else { throw ModelCatalogError.invalidArtifact(artifact.id) }
        }
        var demos = [
            DemoDescriptor(
                id: "comparison", title: "Comparison demo", systemImage: "rectangle.split.3x1",
                summary: "Compare product specifications from source text.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Add up to four product descriptions from URLs or pasted text, then compare extracted specifications and their source evidence."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["comparison"]),
            DemoDescriptor(
                id: "extraction", title: "Extraction demo", systemImage: "text.magnifyingglass",
                summary: "Extract structured fields and review their evidence.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Enter a note and inspect the extracted text spans and scores for each field alongside the original text."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["extraction"]),
            DemoDescriptor(
                id: "inspection", title: "Inspection demo", systemImage: "checklist",
                summary: "Review truck inspection answers entered by text or voice.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Enter inspection answers, review the interpreted fields, and correct them before sharing the inspection record."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["inspection"]),
            DemoDescriptor(
                id: "image-search", title: "Image search demo", systemImage: "photo.badge.magnifyingglass",
                summary: "Find selected images using a text query.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Select images, prepare their search index, and compare the matching images for your query. Similarity scores help rank candidates."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["image-search"]),
            DemoDescriptor(
                id: "video-search", title: "Video search demo", systemImage: "film.stack",
                summary: "Find moments in a video using a text query.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Choose a video, prepare its frame index, and jump to the moments that match your words. Similarity scores help rank candidates."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["video-search"]),
            DemoDescriptor(
                id: "object-finder", title: "Camera search demo", systemImage: "viewfinder",
                summary: "Find camera regions that match your words.",
                explanation: [
                    DemoExplanation(
                        title: "How it works",
                        text:
                            "Enter what you want to find and inspect matching regions in the camera preview. Matches are similarity candidates and may need human review."
                    )
                ],
                platforms: [.iOS], searchTerms: searchTerms["object-finder"]),
        ]
        let glinerFP32 = PublishedModelArtifacts.glinerFP32(platforms: [.iOS])
        let embeddingFP16 = PublishedModelArtifacts.embeddingFP16(platforms: [.iOS])
        let embeddingPTQ = PublishedModelArtifacts.embeddingPTQ(platforms: [.iOS])
        var models = [
            LibraryModel(
                id: "gliner2.5-multi", name: "GLiNER2.5-multi", category: "Information Extraction",
                searchTerms: searchTerms["gliner2.5-multi"] ?? [], artifacts: [glinerFP32],
                defaultArtifactID: glinerFP32.id, platforms: [.iOS],
                demos: [
                    ModelDemoLink(demoID: "comparison", platforms: [.iOS], optionalCapabilities: [.network]),
                    ModelDemoLink(demoID: "extraction", platforms: [.iOS]),
                    ModelDemoLink(
                        demoID: "inspection", platforms: [.iOS],
                        optionalCapabilities: [.microphone, .speechRecognition]),
                ]),
            LibraryModel(
                id: "embeddinggemma2", name: "EmbeddingGemma 2", category: "Multimodal Embedding",
                searchTerms: searchTerms["embeddinggemma2"] ?? [], artifacts: [embeddingFP16, embeddingPTQ],
                defaultArtifactID: embeddingFP16.id, platforms: [.iOS],
                demos: [
                    ModelDemoLink(demoID: "image-search", platforms: [.iOS]),
                    ModelDemoLink(demoID: "video-search", platforms: [.iOS]),
                    ModelDemoLink(demoID: "object-finder", platforms: [.iOS], requiredCapabilities: [.camera]),
                ]),
        ]
        if let initial = images.first {
            demos.append(
                DemoDescriptor(
                    id: "image-generation", title: "Image generation demo", systemImage: "photo.badge.plus",
                    summary:
                        "Create an image from a prompt or edit one reference image. Generation takes several minutes.",
                    explanation: [
                        DemoExplanation(
                            title: "How it works",
                            text: "Generate 512-square images from a prompt, or provide one reference image to edit."),
                        DemoExplanation(
                            title: "1. Prepare the model",
                            text:
                                "Import the complete prepared Qwen Image folder from Files. It uses 13.78 GB, plus space for the app copy. This build does not provide a model download."
                        ),
                        DemoExplanation(
                            title: "2. Describe the image",
                            text:
                                "Enter a prompt. To edit an image, choose one reference and confirm the square crop. Keep the same seed to repeat the same request."
                        ),
                        DemoExplanation(
                            title: "3. Generate and save",
                            text:
                                "Generation takes several minutes. Keep the app open. Stop waits for the current operation to finish. Save or share the completed PNG."
                        ),
                        DemoExplanation(
                            title: "Device and model requirements",
                            text:
                                "This build enables Qwen Image on iPhone 17 Pro with iOS 27.0.1. Check the included LICENSE and NOTICE for the model terms. Your prompt and reference image stay on the device."
                        ),
                    ], platforms: [.iOS], searchTerms: searchTerms["image-generation"]))
            models.append(
                LibraryModel(
                    id: "qwen-image21-turbo", name: "Qwen Image 2.1 Turbo", category: "Image Generation",
                    searchTerms: searchTerms["qwen-image21-turbo"] ?? [], artifacts: images,
                    defaultArtifactID: initial.id, platforms: [.iOS],
                    demos: [ModelDemoLink(demoID: "image-generation", platforms: [.iOS])]))
        }
        for (value, contract) in [
            (speechArtifact, "irodori-coreai-v1"), (referenceSpeechArtifact, "irodori-coreai-v2"),
        ] {
            if let artifact = value {
                guard artifact.contract == contract, artifact.isSourceValid,
                    artifact.localBundle?.manifestPath == "irodori.json" || artifact.download != nil
                else { throw ModelCatalogError.invalidArtifact(artifact.id) }
            }
        }
        let speechArtifacts = [referenceSpeechArtifact, speechArtifact].compactMap { $0 }
        if let initialSpeech = speechArtifacts.first {
            demos.append(
                DemoDescriptor(
                    id: "speech", title: "Speech demo", systemImage: "waveform",
                    summary: referenceSpeechArtifact == nil
                        ? "Turn a short Japanese passage into speech, then play or save it."
                        : "Generate Japanese speech using a recorded voice or an audio file.",
                    explanation: referenceSpeechArtifact.map { referenceExplanation(download: $0.download != nil) } ?? [
                        DemoExplanation(
                            title: "1. Prepare the model",
                            text:
                                "This build does not offer an Irodori model download. In Files, select the complete prepared Irodori FP32 folder with Import model. The folder uses 3.38 GB and is copied into the app, so allow space for both copies and model preparation."
                        ),
                        DemoExplanation(
                            title: "2. Enter what to say",
                            text:
                                "Choose Try an example or enter a short Japanese passage. You can leave the voice variation setting as it is."
                        ),
                        DemoExplanation(
                            title: "3. Generate, listen and save",
                            text:
                                "Tap Generate speech and keep the app open. The first generation can take longer while the model is prepared. When it finishes, tap Play or Save WAV. Stop cancels the current operation."
                        ),
                        DemoExplanation(
                            title: "About reference voices",
                            text:
                                "The current model bundle generates speech from text without a reference voice. Recording, importing a voice recording and voice cloning are not supported by this bundle. Changing the seed changes the generated voice variation."
                        ),
                    ],
                    platforms: [.iOS], searchTerms: searchTerms["speech"]))
            models.append(
                LibraryModel(
                    id: "irodori", name: "Irodori", category: "Speech Synthesis",
                    searchTerms: searchTerms["irodori"] ?? [], artifacts: speechArtifacts,
                    defaultArtifactID: initialSpeech.id, platforms: [.iOS],
                    demos: [ModelDemoLink(demoID: "speech", platforms: [.iOS])]))
        }
        if let chronosArtifact {
            models.append(
                ChronosDemoCatalog.model(artifact: chronosArtifact, platforms: [.iOS], searchTerms: searchTerms))
            demos.append(ChronosDemoCatalog.demo(platforms: [.iOS], searchTerms: searchTerms))
        }
        return try ModelCatalog(models: models, demos: demos)
    }

    private static func referenceExplanation(download: Bool) -> [DemoExplanation] {
        [
            DemoExplanation(
                title: "1. Prepare the model",
                text: download
                    ? "Download the Reference voice model. It uses about 3.84 GB. Wait for the model check to finish before recording or choosing an audio file."
                    : "Import the prepared Reference voice model folder from Files. It uses about 3.84 GB, plus space for the app copy. The legacy model supports text-only generation."
            ),
            DemoExplanation(
                title: "2. Choose a reference voice",
                text:
                    "Tap Record voice, speak clearly, then Finish recording. Or tap Choose audio file and select a WAV, M4A, MP3 or AIFF file. Use 0.5 to 30 seconds of your own voice or a voice you have permission to use. Listen to reference lets you check the selected audio."
            ),
            DemoExplanation(
                title: "3. Enter what to say",
                text:
                    "Choose Try an example or enter a short Japanese passage. Leave the voice options unchanged to begin. A reference voice is optional; leaving it empty uses a generated voice."
            ),
            DemoExplanation(
                title: "4. Generate, listen and save",
                text:
                    "Tap Generate speech and keep the app open. The first run prepares the model and can take longer. Then use Play or Save WAV. Stop cancels recording, reading, generation or playback."
            ),
            DemoExplanation(
                title: "On-device generation",
                text:
                    "Speech generation runs on this device. The app does not upload your reference audio. Saving or sharing a WAV uses the destination you choose."
            ),
        ]
    }

    private static let verifiedImageArtifacts: [ModelArtifact] = [QwenImage21Artifact.nativeComplex]
}
