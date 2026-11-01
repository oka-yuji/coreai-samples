public enum QwenImage21Artifact {
    public static let profileID = "qmi-mobile-w4-dit-g16-native-complex-blocks16-23-f32-v1"
    public static let manifestSHA256 = "8b9e34748d760fa9ec7762f5064652281c80effb19c692bcdbe8ef1e521d59ad"

    public static let nativeComplex = ModelArtifact(
        id: profileID, title: "W4/W8", contract: "qwen-image21-mobile-v2", platforms: [.iOS],
        sizeBytes: 13_783_495_099,
        upstream: ModelUpstream(
            repository: "Qwen/Qwen-Image-2.1-Turbo",
            revision: "fc13ba1986fcb905854dc8de2f52eee395f49b5b"),
        files: [
            ModelArtifactFile(
                path: "LICENSE", bytes: 7_831,
                sha256: "8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d"),
            ModelArtifactFile(
                path: "NOTICE.txt", bytes: 663,
                sha256: "d6aa4e607dfe3aa2b3581aa981be6c15123963d50c50bef4268b3eaabc713e65"),
            ModelArtifactFile(
                path: "README.txt", bytes: 559,
                sha256: "ae8200a7e3ded6eb00b4db0ce196f4a8206e1ebb0f4cf0b5af6f741bdbca678a"),
            ModelArtifactFile(
                path: "bundle.json", bytes: 23_018,
                sha256: "8b9e34748d760fa9ec7762f5064652281c80effb19c692bcdbe8ef1e521d59ad"),
            ModelArtifactFile(
                path: "compression.json", bytes: 2_525,
                sha256: "773d25c01c890904e6450af61870e53f4a099a4482b60f1871d228303e760189"),
            ModelArtifactFile(
                path: "config/processor/chat_template.jinja", bytes: 5_292,
                sha256: "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"),
            ModelArtifactFile(
                path: "config/processor/processor_config.json", bytes: 1_300,
                sha256: "14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b"),
            ModelArtifactFile(
                path: "config/processor/tokenizer.json", bytes: 11_422_650,
                sha256: "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
            ModelArtifactFile(
                path: "config/processor/tokenizer_config.json", bytes: 705,
                sha256: "fd42a0b35a8d13cd473e88e310b50cbf1bccaa9eb60ed0fd873af5f8f0c59307"),
            ModelArtifactFile(
                path: "config/scheduler/scheduler_config.json", bytes: 486,
                sha256: "16c948c71f9152a34cce6e3a2f309e97ad2dca32beecbe5e2fd3a7925b7eb4bb"),
            ModelArtifactFile(
                path: "config/text_encoder/config.json", bytes: 1_542,
                sha256: "89050cba6b66c4f68f52305b54fc37efe28d95232246b94aa8dc0cfa36f3b734"),
            ModelArtifactFile(
                path: "config/transformer/config.json", bytes: 370,
                sha256: "2c567038ca190824728844b8d06d94ae02360e668707afd9734e789a7eb58ce3"),
            ModelArtifactFile(
                path: "config/vae/config.json", bytes: 2_172,
                sha256: "42bfdec412459bbb7bc38a0617ab70edcfc873f84ec47e0c7d01c53ba09885a5"),
            ModelArtifactFile(
                path: "denoiser/component.json", bytes: 18_822,
                sha256: "627197fe7807e3f8509f4b5a69a4892ac05c89aa3a9a34d0e76f40761c8d39dc"),
            ModelArtifactFile(
                path: "denoiser/compression.json", bytes: 227_787,
                sha256: "3c08efbdf1a924c0487c6bd6c8accee05c01fa2f5e67afa4c8f2936c9bbbf396"),
            ModelArtifactFile(
                path: "denoiser/denoiser.aimodel/main.hash", bytes: 32,
                sha256: "1cfa331fe79a115f4c444ece2e5322f11a8211daff315474f01520d319de9012"),
            ModelArtifactFile(
                path: "denoiser/denoiser.aimodel/main.mlirb", bytes: 6_846_755_894,
                sha256: "eacded0298a54afb1980c329c414e53e0b954b79ef905696712b310166eca568"),
            ModelArtifactFile(
                path: "denoiser/denoiser.aimodel/metadata.json", bytes: 567,
                sha256: "96829143670e8053c63ea03d5eecca2eaaaac219a3a83c56018f65736088f254"),
            ModelArtifactFile(
                path: "evidence/A0/identity.json", bytes: 1_074,
                sha256: "dda0719aa91ed20ea6080e310028ecc061ae2897579b187cafe345ad8277ae27"),
            ModelArtifactFile(
                path: "evidence/A0/summary.json", bytes: 166_482,
                sha256: "8e6ffd1928e6ef2a56596a3d81ff6a83e6ae56a5b849b35cdfae2a1f6378a62b"),
            ModelArtifactFile(
                path: "evidence/A1/identity.json", bytes: 15_423,
                sha256: "3a25a23ddc781d3258854f870606e3f7291bbbd1d8402da27d1b641eaee7bde0"),
            ModelArtifactFile(
                path: "evidence/A1/summary.json", bytes: 723_975,
                sha256: "e945bff716ccd31290df22aa9170e20b2ee3d9f7ef46f06d1f103fcc25e54f1b"),
            ModelArtifactFile(
                path: "evidence/B/identity.json", bytes: 722_904,
                sha256: "999e04becad7b256aa28b60be49d03bfb4132a9e54df2398e14d74fdb09e7660"),
            ModelArtifactFile(
                path: "evidence/B/summary.json", bytes: 1_481_128,
                sha256: "4acaf90594df20b194d68299b77200a9eb95019c8a6f617e66e37f674814230d"),
            ModelArtifactFile(
                path: "evidence/comparison.json", bytes: 211_924,
                sha256: "287a0f7ed409678c0dcbe69d4fd1aefdf736e093d841638d9c8a4014d3216a28"),
            ModelArtifactFile(
                path: "evidence/fixtures.json", bytes: 74_388,
                sha256: "b26820d8a34afd01f4f2c8e1ab2b0b66d951fe38e5b0a6b79104c1f555fe3ee9"),
            ModelArtifactFile(
                path: "evidence/numeric.json", bytes: 3_118_784,
                sha256: "d5fc29ccb5f54e62df68497f1b5f31d27890ca47c496e4244e1d39d55c83cd66"),
            ModelArtifactFile(
                path: "evidence/owner-review.json", bytes: 5_564,
                sha256: "a6d76c7224d7807bf364de32aa0fe77772d7e4604829865e17c92fb7e25e5c53"),
            ModelArtifactFile(
                path: "evidence/quality-ratings.json", bytes: 42_107,
                sha256: "da3139c5a6e007001ea95b7c3af66ed52d0b3a3aeaf4ef4532a5bb066844bc1c"),
            ModelArtifactFile(
                path: "evidence/visual-equivalence.json", bytes: 135_961,
                sha256: "c5949123d82e9524e7867a1a5737c29ff3c898fdf2ad0faa5f147c84107b57de"),
            ModelArtifactFile(
                path: "text/component.json", bytes: 18_196,
                sha256: "9b763055751e1a54c8b337a534b3a6a677d1c8b615eb2a29137e66d1d4c3563d"),
            ModelArtifactFile(
                path: "text/compression.json", bytes: 292_373,
                sha256: "336e3ccd64f8e33bb1c86e023bb8a56a45316ef1f4fa731d80dfd92caffda012"),
            ModelArtifactFile(
                path: "text/text.aimodel/main.hash", bytes: 32,
                sha256: "622aa59fd7ce0fb6c7c20fba2c6621d2777690799866e64aec61f398e4bcb4ed"),
            ModelArtifactFile(
                path: "text/text.aimodel/main.mlirb", bytes: 4_936_279_428,
                sha256: "c3637bd1c1254df244cd8511f1b6a71a64b058b0b3bde42c9c1cd77402b79205"),
            ModelArtifactFile(
                path: "text/text.aimodel/metadata.json", bytes: 511,
                sha256: "679eaff8fbead5af9814a8e6460dd2798feaf5493af71b2644fcf3cd523e1d76"),
            ModelArtifactFile(
                path: "uv.lock", bytes: 121_801,
                sha256: "893377255e8793ea9c7cec2d914293a05efd7f1439a6c6783670e55254be1779"),
            ModelArtifactFile(
                path: "vae_decoder/component.json", bytes: 17_176,
                sha256: "9f1612e97b3ab6d2424ac64a4e530a3ff4edd05d6e0dd51c5b91e20edca77157"),
            ModelArtifactFile(
                path: "vae_decoder/compression.json", bytes: 217,
                sha256: "1bbe6ac0cb507d113464d3741344b5474fd50e702d1dc07d0ff2baf9b7306776"),
            ModelArtifactFile(
                path: "vae_decoder/vae_decoder.aimodel/main.hash", bytes: 32,
                sha256: "9c3f029dfabdd506b00b94625d16642e7607ce42c09924b416f48abbd61b79a9"),
            ModelArtifactFile(
                path: "vae_decoder/vae_decoder.aimodel/main.mlirb", bytes: 1_012_367_442,
                sha256: "d7ecc7a9335e4f613436afd9ce3caf0553779db5b2c0a407a5b5f64d2094ffa4"),
            ModelArtifactFile(
                path: "vae_decoder/vae_decoder.aimodel/metadata.json", bytes: 518,
                sha256: "e499fb1e1e498cbe539d600efabf590bb10c6d439e7ad95c61d84b9c5315b00d"),
            ModelArtifactFile(
                path: "vae_encoder/component.json", bytes: 17_174,
                sha256: "7126039244cd6563a00aa3dee98848bda340638891c864f889ad6876ff38e09f"),
            ModelArtifactFile(
                path: "vae_encoder/compression.json", bytes: 217,
                sha256: "e2c30e8b1c472cb3a7593dc06d0deb2e274332656fda768d6fb093e3a598892e"),
            ModelArtifactFile(
                path: "vae_encoder/vae_encoder.aimodel/main.hash", bytes: 32,
                sha256: "81c77f8cfa67e1d12d4a72cf16d95ee5b6f079d80cd96580673ac9d9f982c125"),
            ModelArtifactFile(
                path: "vae_encoder/vae_encoder.aimodel/main.mlirb", bytes: 311_722_097,
                sha256: "3634781357be061a962f235dae4fada2734f5be57d8f91228f03313141361e9a"),
            ModelArtifactFile(
                path: "vae_encoder/vae_encoder.aimodel/metadata.json", bytes: 518,
                sha256: "5dbc6c20ddf0ad59abb20d3207802d4271196491538d743c4c86c1e0efa88d5a"),
            ModelArtifactFile(
                path: "vision/component.json", bytes: 17_712,
                sha256: "1f2775134330ef2c957c4aabcaf5159bcb839e133ba3a7e12247ed8d41c38b49"),
            ModelArtifactFile(
                path: "vision/compression.json", bytes: 134_033,
                sha256: "e2bb96e420e36a719417d5a404068d8d601e1552996de2600858189f9b186f36"),
            ModelArtifactFile(
                path: "vision/vision.aimodel/main.hash", bytes: 32,
                sha256: "871b257311f5556dc8be2fd3e6731d5310cb38bd77b7cdd113cb80f5bafaa737"),
            ModelArtifactFile(
                path: "vision/vision.aimodel/main.mlirb", bytes: 657_333_116,
                sha256: "f56b0e89184d07c001044ef1e35b690e94c264bc04e912b1591140032bad62cb"),
            ModelArtifactFile(
                path: "vision/vision.aimodel/metadata.json", bytes: 513,
                sha256: "25e91c8005adb22f4bd9cd1e353dc6a8a7c422f041d1eb25e7162c56449f57f0"),
        ],
        localBundle: ModelLocalBundle(manifestSHA256: manifestSHA256, profileID: profileID))
}
