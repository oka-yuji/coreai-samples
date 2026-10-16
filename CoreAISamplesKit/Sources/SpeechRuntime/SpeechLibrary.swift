#if os(macOS) || os(iOS)
    import Foundation
    import SpeechFeature

    public protocol SpeechFolderListing: Actor {
        func folders() async throws -> [SpeechFolder]
    }

    public actor SpeechLibrary: SpeechFolderListing {
        private let directory: URL
        public init(directory: URL) { self.directory = directory }
        public func folders() throws -> [SpeechFolder] {
            try Task.checkCancellation()
            let values = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
                options: [.skipsHiddenFiles])
            var folders: [SpeechFolder] = []
            for url in values {
                try Task.checkCancellation()
                let properties = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                guard properties.isDirectory == true, properties.isSymbolicLink != true else { continue }
                let marker = url.appendingPathComponent("irodori.json")
                guard FileManager.default.fileExists(atPath: marker.path) else { continue }
                folders.append(SpeechFolder(url: url))
            }
            return folders.sorted { $0.name < $1.name }
        }
    }

    @MainActor
    public final class SpeechLibraryRunner {
        private let library: any SpeechFolderListing
        public init(library: any SpeechFolderListing) { self.library = library }
        public func run(_ effect: SpeechLibraryEffect, send: (SpeechLibraryAction) -> Void) async {
            let session: Int
            switch effect {
            case .load(let value): session = value
            }
            do {
                let folders = try await library.folders()
                try Task.checkCancellation()
                send(.loaded(session, folders))
            } catch {
                if !Task.isCancelled && !(error is CancellationError) {
                    send(.failed(session, error.localizedDescription))
                }
            }
        }
    }
#endif
