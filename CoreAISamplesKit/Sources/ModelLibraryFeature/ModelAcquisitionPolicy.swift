public struct ModelAcquisitionPolicy: Equatable, Sendable {
    public enum Method: Equatable, Sendable {
        case download, importBundle

        public var progressTitle: String {
            self == .download ? "Downloading" : "Importing"
        }

        public var cancellationTitle: String {
            self == .download ? "Cancel download" : "Cancel import"
        }

        public func actionTitle(retrying: Bool) -> String {
            self == .importBundle ? "Import model" : retrying ? "Retry download" : "Download model"
        }

        public func systemImage(retrying: Bool) -> String {
            self == .importBundle ? "folder.badge.plus" : retrying ? "arrow.clockwise" : "arrow.down.circle"
        }
    }

    public let preferredMethod: Method?
    public let allowsImport: Bool
    public let readyTitle: String

    public init(artifact: ModelArtifact) {
        allowsImport = artifact.isSourceValid && artifact.localBundle != nil
        preferredMethod =
            if artifact.isSourceValid, artifact.download != nil { .download } else if allowsImport {
                .importBundle
            } else { nil }
        readyTitle =
            if artifact.localBundle != nil, artifact.download != nil {
                "Model ready"
            } else if artifact.localBundle != nil { "Imported" } else { "Downloaded" }
    }

    public static func transferMethod(for phase: ModelInstallationPhase) -> Method? {
        switch phase {
        case .downloading: .download
        case .importing: .importBundle
        default: nil
        }
    }
}
