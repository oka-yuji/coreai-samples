import ChronosCore
import Darwin
import Foundation

public typealias ChronosBundleIdentity = ChronosCore.ChronosBundleIdentity

public struct ChronosBundle: Sendable {
    public let url: URL
    public let identity: ChronosBundleIdentity
    public let reference: ChronosReference

    @concurrent
    public static func open(_ url: URL) async throws -> Self {
        try Task.checkCancellation()
        let directory = try ChronosFiles.openDirectory(url)
        defer { Darwin.close(directory) }
        try validateContents(directory)
        let data = try ChronosFiles.read("chronos2.json", in: directory, limit: ChronosFiles.manifestLimit)
        let manifest = try ChronosJSON.decodeExact(ChronosManifest.self, from: data)
        try manifest.validate()
        for name in manifest.files.keys.sorted() {
            guard let expected = manifest.files[name] else { throw ChronosError.invalidBundle("inventory") }
            let values = try url.appendingPathComponent(name).resourceValues(forKeys: [.isAliasFileKey])
            guard values.isAliasFile != true else { throw ChronosError.invalidBundle("alias file") }
            try ChronosFiles.hash(name, in: directory, expected: expected)
        }
        let referenceData = try ChronosFiles.read("reference.json", in: directory, limit: ChronosFiles.referenceLimit)
        let verificationData = try ChronosFiles.read(
            "verification.json", in: directory, limit: ChronosFiles.manifestLimit)
        guard ChronosFiles.identity(referenceData) == manifest.files["reference.json"],
            ChronosFiles.identity(verificationData) == manifest.files["verification.json"]
        else { throw ChronosError.invalidBundle("reference changed") }
        let reference = try ChronosJSON.decodeExact(ChronosReference.self, from: referenceData)
        try reference.validate()
        try ChronosJSON.decodeExact(ChronosPriorVerification.self, from: verificationData).validate(
            reference: reference)
        guard try ChronosFiles.read("chronos2.json", in: directory, limit: ChronosFiles.manifestLimit) == data else {
            throw ChronosError.invalidBundle("manifest changed")
        }
        try validateContents(directory)
        try Task.checkCancellation()
        return Self(
            url: url,
            identity: ChronosBundleIdentity(
                source: manifest.source, manifest: ChronosFiles.identity(data), files: manifest.files),
            reference: reference)
    }

    private static func validateContents(_ directory: Int32) throws {
        let expected = Set(ChronosManifest.fileNames.map { String($0.split(separator: "/")[0]) })
            .union(["chronos2.json"])
        guard try ChronosFiles.names(in: directory) == expected else {
            throw ChronosError.invalidBundle("missing or extra bundle entries")
        }
        let model = openat(directory, "Model.aimodel", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard model >= 0 else { throw ChronosFiles.failure() }
        defer { Darwin.close(model) }
        guard try ChronosFiles.names(in: model) == ["main.hash", "main.mlirb", "metadata.json"] else {
            throw ChronosError.invalidBundle("missing or extra asset entries")
        }
    }
}
