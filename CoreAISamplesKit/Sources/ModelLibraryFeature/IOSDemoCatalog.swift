public enum IOSDemoCatalog {
    public static func make(searchTerms: [String: [String]] = [:]) throws -> ModelCatalog {
        let demos = [
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
        let models = [
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
        return try ModelCatalog(models: models, demos: demos)
    }

    private static func artifact(
        _ id: String, title: String, contract: String, download: ModelDownload, upstream: ModelUpstream,
        files: [ModelArtifactFile]
    ) -> ModelArtifact {
        ModelArtifact(
            id: id, title: title, contract: contract, platforms: [.iOS], sizeBytes: files.reduce(0) { $0 + $1.bytes },
            download: download, upstream: upstream, files: files)
    }

    private static let glinerFP32 = artifact(
        "gliner2.5-multi-fp32-t256-t512", title: "FP32", contract: "gliner-gl4-fp32-t256-t512",
        download: ModelDownload(
            repository: "okayuji/gliner2.5-multi-coreai-fp32", revision: "043e738f3a91222ec013b2d80e240e34abb46bdb"),
        upstream: ModelUpstream(
            repository: "fastino/gliner2.5-multi-v1", revision: "2ca71aafb3446d9014e1c55c7ff51c9bc7209c47"),
        files: [
            ModelArtifactFile(
                path: "contract.json", bytes: 2_204,
                sha256: "e83638c3288ce64a6ed446078577922c1961f2df5842d76fc5a4bfb2598c3860"),
            ModelArtifactFile(
                path: "gliner25_multi_fp32_t256_t512.aimodel/main.hash", bytes: 32,
                sha256: "0a2d223eb8527e3085a78b0c7f66df299b9d76e5d2c0f59d85d0d9e46e340308"),
            ModelArtifactFile(
                path: "gliner25_multi_fp32_t256_t512.aimodel/main.mlirb", bytes: 1_231_651_630,
                sha256: "e37ad7f9a2ecb98323cd28bc40d1a9efa733c87a70ee7dfcf1cab31ef0bce1dc"),
            ModelArtifactFile(
                path: "gliner25_multi_fp32_t256_t512.aimodel/metadata.json", bytes: 2_417,
                sha256: "9f64e4404d34926d62cdc7d71f0b740669ae5128bc6b71a08f1eaad37641cf13"),
            ModelArtifactFile(
                path: "tokenizer/tokenizer.json", bytes: 16_035_853,
                sha256: "c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a"),
            ModelArtifactFile(
                path: "LICENSE", bytes: 11_358,
                sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
            ModelArtifactFile(
                path: "THIRD_PARTY_NOTICES.md", bytes: 4_190,
                sha256: "f73981af85975e48c271db74816b939b9f5505ede4c018430f2a3979ed8c76f8"),
            ModelArtifactFile(
                path: "README.md", bytes: 25_702,
                sha256: "9777d6aa5fd0a7a95f12d1dbac3f127c0d54269dea095b875520cfc2651ef823"),
        ])

    private static let embeddingRepository = "okayuji/embeddinggemma-2-coreai"
    private static let embeddingRevision = "daab6dbfbae1610a87d9f181458419e3ea5ce682"
    private static let embeddingUpstream = ModelUpstream(
        repository: "google/embeddinggemma-2", revision: "914f7f89142e33e77833254d9c9b90c3cef7303b")

    private static let embeddingFP16 = artifact(
        "embeddinggemma2-fp16-safe-s128-v70", title: "FP16", contract: "embeddinggemma2-fp16-safe-s128-v70",
        download: ModelDownload(
            repository: embeddingRepository, revision: embeddingRevision, root: "fp16"),
        upstream: embeddingUpstream,
        files: [
            ModelArtifactFile(
                path: "embedding-manifest.json", bytes: 1_279,
                sha256: "1a1122d998d8af9ce83c789771fe7f5f6946e181ca87572104d0a95892e65c04"),
            ModelArtifactFile(
                path: "LICENSE", bytes: 11_358,
                sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
            ModelArtifactFile(
                path: "MODEL_LICENSE.txt", bytes: 429,
                sha256: "d0fa0ee7cc36c8418c95f92f1f1d4a5bbba43a69f0009bba0765b4544297d9f9"),
            ModelArtifactFile(
                path: "README.md", bytes: 21_730,
                sha256: "b677a0ee2818c36f091b65fd3d9b4baea9f04d53b02b49cae916c74bf9682037"),
            ModelArtifactFile(
                path: "text.aimodel/main.hash", bytes: 32,
                sha256: "733a0bebd7dea5501ae2c521dde8e47c448cbba80740dd2c60f452f157015ca1"),
            ModelArtifactFile(
                path: "text.aimodel/main.mlirb", bytes: 277_237_425,
                sha256: "6c33980ae6f10887d10210d4b0f7a18804cdc18ea737e38de898ae7eafdc1a25"),
            ModelArtifactFile(
                path: "text.aimodel/metadata.json", bytes: 408,
                sha256: "9688fbfe65a79b4992b2e98f70fc74150b836c012105fc906bad01e1636e7c07"),
            ModelArtifactFile(
                path: "token_embeddings.bf16", bytes: 268_435_456,
                sha256: "56ebbdbfd706c827f3fae1d97aa39135c7ddc0f1e6d9b12f8565f84bbfbda877"),
            ModelArtifactFile(
                path: "tokenizer.json", bytes: 32_170_510,
                sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
            ModelArtifactFile(
                path: "tokenizer_config.json", bytes: 1_599,
                sha256: "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874"),
            ModelArtifactFile(
                path: "vision.aimodel/main.hash", bytes: 32,
                sha256: "8569d81f837a44694b01fc6c57c4847e6a4c906ee041178d8a502735ed77252d"),
            ModelArtifactFile(
                path: "vision.aimodel/main.mlirb", bytes: 335_796_665,
                sha256: "1840476dffb4aba3c53b7dde394ed314cf0d88d7d9e0f7e4eebfede90a046f87"),
            ModelArtifactFile(
                path: "vision.aimodel/metadata.json", bytes: 410,
                sha256: "d033bac7a486e70dfe87029e295ec97e5e4278cfc5005ee533631cf73ee410c4"),
        ])

    private static let embeddingPTQ = artifact(
        "embeddinggemma2-int4-int8-ptq-s128-v70", title: "INT4/INT8 PTQ",
        contract: "embeddinggemma2-int4-int8-ptq-s128-v70",
        download: ModelDownload(
            repository: embeddingRepository, revision: embeddingRevision, root: "int4-int8"),
        upstream: embeddingUpstream,
        files: [
            ModelArtifactFile(
                path: "embedding-manifest.json", bytes: 1_586,
                sha256: "7c37ea823af7a4028ec3757e124b53fe8e16a75c99c46d24ca0a7007cca55b85"),
            ModelArtifactFile(
                path: "LICENSE", bytes: 11_358,
                sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
            ModelArtifactFile(
                path: "MODEL_LICENSE.txt", bytes: 549,
                sha256: "e01569a50186d4bb7b61a86f729534796d2ed5f62db83fe58369136f8b94c210"),
            ModelArtifactFile(
                path: "README.md", bytes: 21_730,
                sha256: "b677a0ee2818c36f091b65fd3d9b4baea9f04d53b02b49cae916c74bf9682037"),
            ModelArtifactFile(
                path: "quantization-quality.json", bytes: 99_783,
                sha256: "57386ed652cf1bc5da3727d8e8228da926d9536ac862533b8bd56da660f3eafe"),
            ModelArtifactFile(
                path: "quantized-reference.json", bytes: 341_671,
                sha256: "fab7750b30160cf9c6cf2b3725578a9b423916b1f5293360ed9cc705e1ee3b91"),
            ModelArtifactFile(
                path: "text.aimodel/main.hash", bytes: 32,
                sha256: "18ba6e8359d4be013f5d35433ed226872802b0ae0002c57e6c913bcfcb593531"),
            ModelArtifactFile(
                path: "text.aimodel/main.mlirb", bytes: 72_639_389,
                sha256: "a041adb63ba236503ce00246700363a538b6d0104bc6aaaa4ce088aaf896dca0"),
            ModelArtifactFile(
                path: "text.aimodel/metadata.json", bytes: 408,
                sha256: "400fdc655ad90d68dbec87c2e023da3c81a4c7604c1b62d4f0166835ff228ef2"),
            ModelArtifactFile(
                path: "token_embedding_scales.f32", bytes: 1_048_576,
                sha256: "6ec75cbc11148befc45977a9af84021cce1fc6b90abe3bc3c46450f20549202c"),
            ModelArtifactFile(
                path: "token_embeddings.int4", bytes: 67_108_864,
                sha256: "a019c72a06e49637fe3142e13e0a2bd8386743dfbaa6e3b57596837e7a17c502"),
            ModelArtifactFile(
                path: "tokenizer.json", bytes: 32_170_510,
                sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
            ModelArtifactFile(
                path: "tokenizer_config.json", bytes: 1_599,
                sha256: "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874"),
            ModelArtifactFile(
                path: "vision.aimodel/main.hash", bytes: 32,
                sha256: "2867db64b2b4f76577f4a44251f4d97a96bcf71268eb2b111214b6dfdbd880c0"),
            ModelArtifactFile(
                path: "vision.aimodel/main.mlirb", bytes: 168_433_544,
                sha256: "dd750cbf7a327f3aa6b3c4d07ea88d5601153809483a8a18789ca6bc83540e92"),
            ModelArtifactFile(
                path: "vision.aimodel/metadata.json", bytes: 410,
                sha256: "d8072e1ac94d3de5eb9d9d06a5e81c8252f9cab82d316fa2af93963f30634ea2"),
        ])
}
