#if os(macOS) || os(iOS)
    import Darwin
    import Foundation
    import SpeechCore

    public actor SpeechCheckFiles {
        private let documents: URL
        private var output: URL?
        public init(documents: URL) { self.documents = documents }
        public static func validName(_ name: String) -> Bool {
            let bytes = Array(name.utf8)
            func alphanumeric(_ byte: UInt8) -> Bool {
                (48...57).contains(byte) || (65...90).contains(byte) || (97...122).contains(byte)
            }
            guard let first = bytes.first, bytes.count <= 80, alphanumeric(first) else { return false }
            return bytes.allSatisfy { alphanumeric($0) || $0 == 45 || $0 == 95 }
        }
        public func reserve(_ runID: String) throws -> URL {
            output = nil
            guard Self.validName(runID) else { throw SpeechError.invalidAsset("Use a unique alphanumeric check ID.") }
            try Task.checkCancellation()
            let root = documents.appendingPathComponent("SpeechChecks", isDirectory: true)
            guard !FileManager.default.fileExists(atPath: root.appendingPathComponent("irodori.json").path) else {
                throw SpeechError.invalidAsset(
                    "SpeechChecks is reserved for diagnostics. Move the model to another folder.")
            }
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            guard try root.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
                throw SpeechError.invalidAsset("The check output folder must not be a symbolic link.")
            }
            let directory = root.appendingPathComponent(runID, isDirectory: true)
            guard mkdir(directory.path, 0o700) == 0 else {
                throw NSError(
                    domain: NSPOSIXErrorDomain, code: Int(errno), userInfo: [NSFilePathErrorKey: directory.path])
            }
            output = directory
            return directory
        }
        public func modelFolder(_ name: String) throws -> URL {
            guard Self.validName(name), name.caseInsensitiveCompare("SpeechChecks") != .orderedSame else {
                throw SpeechError.invalidAsset("Select a model folder in Documents.")
            }
            let folder = documents.appendingPathComponent(name, isDirectory: true)
            let resolved = folder.resolvingSymlinksInPath()
            guard resolved.deletingLastPathComponent() == documents.resolvingSymlinksInPath(),
                resolved.lastPathComponent.caseInsensitiveCompare("SpeechChecks") != .orderedSame
            else {
                throw SpeechError.invalidAsset("The model folder must be inside Documents.")
            }
            return folder
        }
        public func writeWAV(_ data: Data, index: Int) throws {
            guard let output else { throw SpeechError.invalidAsset("No check output folder was reserved.") }
            try data.write(to: output.appendingPathComponent("sample-\(index).wav"), options: .withoutOverwriting)
        }
        public func referenceFile(_ name: String) throws -> URL {
            guard !name.isEmpty, name.utf8.count <= 160, !name.contains("/"), !name.contains("\\"), name != ".",
                name != ".."
            else {
                throw SpeechError.invalidAsset("Choose a reference audio file directly inside Documents.")
            }
            let file = documents.appendingPathComponent(name)
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true,
                file.resolvingSymlinksInPath().deletingLastPathComponent() == documents.resolvingSymlinksInPath()
            else { throw SpeechError.invalidAsset("Choose a reference audio file directly inside Documents.") }
            return file
        }
        public func writeReport(_ data: Data) throws {
            guard let output else { throw SpeechError.invalidAsset("No check output folder was reserved.") }
            try data.write(to: output.appendingPathComponent("report.json"), options: .withoutOverwriting)
        }
        public func writeStatus(_ data: Data) throws {
            guard let output else { return }
            try data.write(to: output.appendingPathComponent("status.json"), options: .atomic)
        }
    }
#endif
