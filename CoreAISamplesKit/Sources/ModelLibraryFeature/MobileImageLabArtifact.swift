public enum MobileImageLabArtifact {
    public static let manifestSHA256 = "9d61c180a8c72b67ba34e8ece6e842fec489dc98fc68b2b875fa07a1900dd3e5"

    public static let published = ModelArtifact(
        id: "mobile-image-lab-w4-w8-v1", title: "Mobile Image Lab W4/W8", contract: "qwen-image21-mobile-v2",
        platforms: [.iOS], sizeBytes: 13_783_495_057,
        download: ModelDownload(
            repository: "okayuji/mobile-image-lab-coreai", revision: "fdd2b8cb17f48976207039f1f780360ba60275dc",
            root: "bundle"),
        upstream: QwenImage21Artifact.nativeComplex.upstream,
        files: QwenImage21Artifact.nativeComplex.files.map { file in
            switch file.path {
            case "README.txt":
                return ModelArtifactFile(
                    path: file.path, bytes: 538,
                    sha256: "26383ebfd2abdfbd07f9405a1e0553f1edf22758d753c8109890b837ece2a5f5")
            case "NOTICE.txt":
                return ModelArtifactFile(
                    path: file.path, bytes: 642,
                    sha256: "7b42feee2cf48a78b2ad8051d269ac03f106b8fbb076f132770c80879b2586d1")
            case "bundle.json":
                return ModelArtifactFile(path: file.path, bytes: 23_018, sha256: manifestSHA256)
            default: return file
            }
        },
        localBundle: ModelLocalBundle(manifestSHA256: manifestSHA256, profileID: QwenImage21Artifact.profileID))
}
