import CryptoKit
import EmbeddingRuntime
import Foundation
import GLiNERRuntime
import ImageGenerationCore
import ModelLibraryFeature
import SpeechAssets

public enum ManagedModelStoreError: Error, Equatable {
    case invalidDefinition
    case pathEscape
    case symbolicLink
    case missingFile(String)
    case unexpectedFile(String)
    case unreadableDirectory(String)
    case sizeMismatch(String)
    case checksumMismatch(String)
    case contractMismatch(String)
    case inUse
    case notInstalled
    case ledgerCorrupted
    case insufficientSpace
}

public enum ManagedInspection: Sendable {
    case ready(ValidatedModelResource)
    case missing
    case invalid(any Error)
}

public enum MigrationOutcome: Sendable {
    case migrated(ValidatedModelResource)
    case skipped
    case undetermined(any Error)
    case failed(any Error)
}

public struct MigrationReport: Sendable {
    public let outcomes: [URL: MigrationOutcome]
}

public struct ResourceLease: Sendable {
    public let identity: ModelArtifactIdentity
    public let owner: UUID
    let id = UUID()
    let store: ManagedModelStore

    public func release() async { await store.release(id) }
}

struct Ledger: Codable {
    struct Entry: Codable {
        let modelID: String
        let artifactID: String
        let contract: String
        let revision: String
        let upstreamRevision: String?
        let sizeBytes: Int64
        let installedAt: Date
        let generation: Int
        let sourceKind: ModelSourceKind?

        init(
            modelID: String, artifactID: String, contract: String, revision: String, upstreamRevision: String?,
            sizeBytes: Int64, installedAt: Date, generation: Int, sourceKind: ModelSourceKind? = nil
        ) {
            self.modelID = modelID
            self.artifactID = artifactID
            self.contract = contract
            self.revision = revision
            self.upstreamRevision = upstreamRevision
            self.sizeBytes = sizeBytes
            self.installedAt = installedAt
            self.generation = generation
            self.sourceKind = sourceKind
        }

        func matches(_ identity: ModelArtifactIdentity) -> Bool {
            modelID == identity.modelID && artifactID == identity.artifactID && contract == identity.contract
                && revision == identity.revision && (sourceKind ?? .huggingFace) == identity.sourceKind
        }

        func records(_ resource: ValidatedModelResource) -> Bool {
            matches(resource.identity) && upstreamRevision == resource.upstreamRevision
                && sizeBytes == resource.sizeBytes
        }
    }

    var format = 1
    var entries: [Entry] = []
}

public actor ManagedModelStore {
    private let root: URL
    private let clock: @Sendable () -> Date
    private let writeLedger: @Sendable (Data, URL) throws -> Void
    private let availableCapacity: @Sendable (URL) throws -> Int64
    private var leases: [UUID: ModelArtifactIdentity] = [:]
    private var generation = 0

    public init(
        root: URL, clock: @escaping @Sendable () -> Date = { .now },
        writeLedger: @escaping @Sendable (Data, URL) throws -> Void = { try $0.write(to: $1, options: .atomic) },
        availableCapacity: @escaping @Sendable (URL) throws -> Int64 = {
            try $0.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
                .volumeAvailableCapacityForImportantUsage ?? .max
        }
    ) {
        self.root = root
        self.clock = clock
        self.writeLedger = writeLedger
        self.availableCapacity = availableCapacity
    }

    public func stagingDirectory(for model: LibraryModel, artifact: ModelArtifact) throws -> URL {
        let resource = try definition(model, artifact)
        let folder = try location("staging", resource.artifactID, resource.revision)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    public func copyForImport(
        from source: URL, model: LibraryModel, artifact: ModelArtifact,
        progress: @Sendable (ModelTransferProgress) -> Void
    ) throws -> URL {
        guard artifact.sourceKind == .localManifest, source.isFileURL else {
            throw ManagedModelStoreError.invalidDefinition
        }
        let resource = try definition(model, artifact)
        guard activeLeases(identity: resource.identity) == 0 else { throw ManagedModelStoreError.inUse }
        let accessed = source.startAccessingSecurityScopedResource()
        defer { if accessed { source.stopAccessingSecurityScopedResource() } }
        let managedPath = root.standardizedFileURL.resolvingSymlinksInPath().path
        let sourcePath = source.standardizedFileURL.resolvingSymlinksInPath().path
        guard sourcePath != managedPath, !sourcePath.hasPrefix(managedPath + "/"),
            !managedPath.hasPrefix(sourcePath + "/")
        else { throw ManagedModelStoreError.pathEscape }
        try verifyStructure(source, artifact, allowingExtraneous: false)
        let area = try stagingDirectory(for: model, artifact: artifact)
        guard try availableCapacity(area) >= resource.sizeBytes else { throw ManagedModelStoreError.insufficientSpace }
        let staging = area.appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        var completed: Int64 = 0
        do {
            let ordered = artifact.files.sorted {
                if ($0.path == artifact.localBundle?.manifestPath) != ($1.path == artifact.localBundle?.manifestPath) {
                    return $0.path == artifact.localBundle?.manifestPath
                }
                return $0.path < $1.path
            }
            for file in ordered {
                try Task.checkCancellation()
                let target = staging.appending(path: file.path)
                try FileManager.default.createDirectory(
                    at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                try copyLocalBundleFile(source: source, file: file, target: target) { increment in
                    completed += increment
                    if let value = ModelTransferProgress(completedBytes: completed, totalBytes: resource.sizeBytes) {
                        progress(value)
                    }
                }
                if file.path == artifact.localBundle?.manifestPath {
                    try verifyLocalManifest(Data(contentsOf: target), artifact)
                }
            }
            try Task.checkCancellation()
            try verifyStructure(source, artifact, allowingExtraneous: false)
            return staging
        } catch {
            try? FileManager.default.removeItem(at: staging)
            _ = rmdir(area.path)
            throw error
        }
    }

    public func inspect(model: LibraryModel, artifact: ModelArtifact) throws -> ManagedInspection {
        guard artifact.isSourceValid else { return .missing }
        let resource: ValidatedModelResource
        do { resource = try definition(model, artifact) } catch { return .invalid(error) }
        let entry = try readLedger()?.entries.first {
            $0.matches(resource.identity)
        }
        if itemType(resource.directory) == nil, entry == nil { return .missing }
        do { try verify(resource.directory, artifact) } catch let error as CancellationError { throw error } catch {
            return .invalid(error)
        }
        if entry?.records(resource) != true { try record(resource) }
        return .ready(resource)
    }

    public func discardImport(staging: URL, model: LibraryModel, artifact: ModelArtifact) throws {
        guard artifact.sourceKind == .localManifest else { throw ManagedModelStoreError.invalidDefinition }
        let resource = try definition(model, artifact)
        let area = try location("staging", resource.artifactID, resource.revision)
        let expected = area.standardizedFileURL.resolvingSymlinksInPath().path + "/"
        let candidate = staging.standardizedFileURL.resolvingSymlinksInPath().path
        guard candidate.hasPrefix(expected), itemType(staging) != .typeSymbolicLink else {
            throw ManagedModelStoreError.pathEscape
        }
        if itemType(staging) != nil { try FileManager.default.removeItem(at: staging) }
        _ = rmdir(area.path)
    }

    public func promote(staging: URL, model: LibraryModel, artifact: ModelArtifact) throws -> ValidatedModelResource {
        let resource = try definition(model, artifact)
        guard activeLeases(identity: resource.identity) == 0 else { throw ManagedModelStoreError.inUse }
        let area = try location("staging", resource.artifactID, resource.revision)
        let areaPath = area.standardizedFileURL.resolvingSymlinksInPath().path
        let stagingPath = staging.standardizedFileURL.resolvingSymlinksInPath().path
        guard stagingPath == areaPath || stagingPath.hasPrefix(areaPath + "/") else {
            throw ManagedModelStoreError.pathEscape
        }
        try verify(staging, artifact)
        try Task.checkCancellation()
        let manager = FileManager.default
        let target = resource.directory
        let trash = try trashItem()
        try manager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        let replacing = itemType(target) != nil
        if replacing { try manager.moveItem(at: target, to: trash) }
        do {
            try manager.moveItem(at: staging, to: target)
            do { try record(resource) } catch {
                try? manager.moveItem(at: target, to: staging)
                throw error
            }
        } catch {
            if replacing { try? manager.moveItem(at: trash, to: target) }
            throw error
        }
        try? manager.removeItem(at: trash)
        _ = rmdir(area.path)
        return resource
    }

    public func migrate(candidates: [URL], model: LibraryModel, artifact: ModelArtifact) -> MigrationReport {
        var outcomes: [URL: MigrationOutcome] = [:]
        guard artifact.sourceKind == .huggingFace else {
            for candidate in candidates {
                outcomes[candidate] = .undetermined(ManagedModelStoreError.invalidDefinition)
            }
            return MigrationReport(outcomes: outcomes)
        }
        var ready = false
        do { if case .ready = try inspect(model: model, artifact: artifact) { ready = true } } catch {
            for candidate in candidates { outcomes[candidate] = .failed(error) }
            return MigrationReport(outcomes: outcomes)
        }
        for candidate in candidates where outcomes[candidate] == nil {
            let outcome = ready ? .skipped : copy(candidate, model: model, artifact: artifact)
            if case .migrated = outcome { ready = true }
            outcomes[candidate] = outcome
        }
        return MigrationReport(outcomes: outcomes)
    }

    public func delete(identity: ModelArtifactIdentity) throws -> Int64 {
        guard activeLeases(identity: identity) == 0 else { throw ManagedModelStoreError.inUse }
        guard var ledger = try readLedger() else { throw ManagedModelStoreError.ledgerCorrupted }
        guard ledger.entries.contains(where: { $0.matches(identity) }) else {
            throw ManagedModelStoreError.notInstalled
        }
        let target = try location("artifacts", identity.artifactID, identity.revision)
        let staging = try location("staging", identity.artifactID, identity.revision)
        let bytes = size(of: target) + size(of: staging)
        ledger.entries.removeAll { $0.matches(identity) }
        try writeLedger(JSONEncoder().encode(ledger), ledgerURL)
        let manager = FileManager.default
        if itemType(target) != nil {
            let trash = try trashItem()
            try manager.moveItem(at: target, to: trash)
            try manager.removeItem(at: trash)
        }
        if itemType(staging) != nil { try manager.removeItem(at: staging) }
        return bytes
    }

    public func deletionSize(identity: ModelArtifactIdentity) -> Int64 {
        guard let target = try? location("artifacts", identity.artifactID, identity.revision),
            let staging = try? location("staging", identity.artifactID, identity.revision)
        else { return 0 }
        return size(of: target) + size(of: staging)
    }

    public func lease(identity: ModelArtifactIdentity, owner: UUID) -> ResourceLease {
        let lease = ResourceLease(identity: identity, owner: owner, store: self)
        leases[lease.id] = identity
        return lease
    }

    public func activeLeases(identity: ModelArtifactIdentity) -> Int {
        leases.values.filter { $0 == identity }.count
    }

    public func generation(of identity: ModelArtifactIdentity) -> Int? {
        (try? readLedger())?.entries.first { $0.matches(identity) }?.generation
    }

    func release(_ id: UUID) { leases[id] = nil }

    private var ledgerURL: URL { root.appending(path: "ledger.json") }

    private func copy(_ candidate: URL, model: LibraryModel, artifact: ModelArtifact) -> MigrationOutcome {
        let base = root.resolvingSymlinksInPath().standardizedFileURL.path
        let source = candidate.resolvingSymlinksInPath().standardizedFileURL.path
        guard source != base, !source.hasPrefix(base + "/") else {
            return .undetermined(ManagedModelStoreError.pathEscape)
        }
        do {
            _ = try definition(model, artifact)
            try verifyFiles(candidate, artifact, allowingExtraneous: true)
        } catch let error as CancellationError { return .failed(error) } catch { return .undetermined(error) }
        let staging: URL
        do {
            staging = try stagingDirectory(for: model, artifact: artifact)
            guard try availableCapacity(staging) >= artifact.files.reduce(0, { $0 + $1.bytes }) else {
                throw ManagedModelStoreError.insufficientSpace
            }
        } catch { return .failed(error) }
        do {
            try FileManager.default.removeItem(at: staging)
            for file in artifact.files {
                try Task.checkCancellation()
                let destination = staging.appending(path: file.path)
                try FileManager.default.createDirectory(
                    at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try FileManager.default.copyItem(at: candidate.appending(path: file.path), to: destination)
            }
            return .migrated(try promote(staging: staging, model: model, artifact: artifact))
        } catch {
            try? FileManager.default.removeItem(at: staging)
            return .failed(error)
        }
    }

    private func definition(_ model: LibraryModel, _ artifact: ModelArtifact) throws -> ValidatedModelResource {
        guard artifact.isSourceValid, let kind = artifact.sourceKind, let revision = artifact.sourceRevision,
            let bytes = artifact.totalFileBytes, model.artifact(artifact.id) == artifact,
            !artifact.files.isEmpty, Set(artifact.files.map(\.path)).count == artifact.files.count
        else { throw ManagedModelStoreError.invalidDefinition }
        guard artifact.download.map({ $0.root.isEmpty || isRelative($0.root) }) ?? true,
            artifact.files.allSatisfy({ isRelative($0.path) })
        else { throw ManagedModelStoreError.pathEscape }
        _ = try family(artifact)
        return ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract, revision: revision,
            upstreamRevision: artifact.upstream?.revision,
            directory: try location("artifacts", artifact.id, revision), sizeBytes: bytes, sourceKind: kind)
    }

    private func readLedger() throws -> Ledger? {
        guard FileManager.default.fileExists(atPath: ledgerURL.path) else { return Ledger() }
        let data = try Data(contentsOf: ledgerURL)
        guard let ledger = try? JSONDecoder().decode(Ledger.self, from: data), ledger.format == 1 else { return nil }
        generation = max(generation, ledger.entries.map(\.generation).max() ?? 0)
        return ledger
    }

    private func record(_ resource: ValidatedModelResource) throws {
        var ledger = try readLedger() ?? Ledger()
        let next = generation + 1
        ledger.entries.removeAll { $0.matches(resource.identity) }
        ledger.entries.append(
            Ledger.Entry(
                modelID: resource.modelID, artifactID: resource.artifactID, contract: resource.contract,
                revision: resource.revision, upstreamRevision: resource.upstreamRevision,
                sizeBytes: resource.sizeBytes, installedAt: clock(), generation: next,
                sourceKind: resource.sourceKind == .huggingFace ? nil : resource.sourceKind))
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try writeLedger(JSONEncoder().encode(ledger), ledgerURL)
        generation = next
    }

    private func location(_ area: String, _ artifactID: String, _ revision: String) throws -> URL {
        guard [artifactID, revision].allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("/") })
        else { throw ManagedModelStoreError.pathEscape }
        let base = root.resolvingSymlinksInPath().standardizedFileURL.path + "/"
        var url = root
        for name in [area, artifactID, revision] {
            url.append(path: name, directoryHint: .isDirectory)
            guard let type = itemType(url) else { continue }
            if type == .typeSymbolicLink { throw ManagedModelStoreError.symbolicLink }
            guard url.resolvingSymlinksInPath().standardizedFileURL.path.hasPrefix(base) else {
                throw ManagedModelStoreError.pathEscape
            }
        }
        return url
    }

    private func trashItem() throws -> URL {
        let folder = root.appending(path: "trash", directoryHint: .isDirectory)
        try? FileManager.default.removeItem(at: folder)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder.appending(path: UUID().uuidString)
    }

    private func size(of directory: URL) -> Int64 {
        guard itemType(directory) == .typeDirectory, let found = try? entries(in: directory) else { return 0 }
        return found.filter { $0.mode == S_IFREG }.reduce(0) { $0 + $1.size }
    }
}

private func isRelative(_ path: String) -> Bool {
    path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
}

private func itemType(_ url: URL) -> FileAttributeType? {
    (try? FileManager.default.attributesOfItem(atPath: url.path))?[.type] as? FileAttributeType
}

private func digest(_ file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file)
    defer { try? handle.close() }
    var hash = SHA256()
    while try autoreleasepool(invoking: {
        guard let data = try handle.read(upToCount: 1_048_576), !data.isEmpty else { return false }
        try Task.checkCancellation()
        hash.update(data: data)
        return true
    }) {}
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}

private enum ContractFamily {
    case gliner
    case embedding(precision: String)
    case image
    case speech
}

private struct ContractSource: Decodable {
    let model: String?
    let revision: String?
}

private func family(_ artifact: ModelArtifact) throws -> ContractFamily {
    if artifact.localBundle != nil {
        if let profile = SpeechBundleProfile(rawValue: artifact.contract) {
            guard artifact == (try IrodoriModelArtifact.make(profile: profile)) else {
                throw ManagedModelStoreError.contractMismatch("Irodori inventory")
            }
            return .speech
        }
        guard artifact.contract == "qwen-image21-mobile-v2", artifact.download == nil,
            artifact.localBundle?.manifestPath == "bundle.json",
            artifact.upstream?.repository == "Qwen/Qwen-Image-2.1-Turbo",
            artifact.upstream?.revision == ImageGenerationProfile.revision,
            Set(["bundle.json", "LICENSE", "NOTICE.txt", "config/processor/tokenizer.json"])
                .isSubset(of: Set(artifact.files.map(\.path)))
        else { throw ManagedModelStoreError.contractMismatch("local image bundle") }
        return .image
    }
    if artifact.contract.hasPrefix("gliner-gl4-") {
        guard artifact.files.contains(where: { ["tokenizer/tokenizer.json", "tokenizer.json"].contains($0.path) })
        else { throw ManagedModelStoreError.contractMismatch("tokenizer") }
        return .gliner
    }
    switch artifact.contract {
    case "embeddinggemma2-fp16-safe-s128-v70": return .embedding(precision: "fp16-safe-s128-v70")
    case "embeddinggemma2-int4-int8-ptq-s128-v70": return .embedding(precision: "int4-int8-ptq-s128-v70")
    default: throw ManagedModelStoreError.contractMismatch("contract")
    }
}

private struct DirectoryEntry {
    let path: String
    let mode: mode_t
    let size: Int64
}

private func entries(in directory: URL) throws -> [DirectoryEntry] {
    var found: [DirectoryEntry] = []
    var pending = [""]
    while let folder = pending.popLast() {
        try Task.checkCancellation()
        let base = folder.isEmpty ? directory.path : directory.path + "/" + folder
        let unreadable = ManagedModelStoreError.unreadableDirectory(folder.isEmpty ? "." : folder)
        guard let handle = opendir(base) else { throw unreadable }
        defer { closedir(handle) }
        while true {
            errno = 0
            guard let entry = readdir(handle) else {
                if errno != 0 { throw unreadable }
                break
            }
            let name = withUnsafeBytes(of: entry.pointee.d_name) {
                String(decoding: $0.prefix(Int(entry.pointee.d_namlen)), as: UTF8.self)
            }
            guard name != ".", name != ".." else { continue }
            var info = stat()
            guard lstat(base + "/" + name, &info) == 0 else { throw unreadable }
            let path = folder.isEmpty ? name : folder + "/" + name
            found.append(DirectoryEntry(path: path, mode: info.st_mode & S_IFMT, size: Int64(info.st_size)))
            if info.st_mode & S_IFMT == S_IFDIR { pending.append(path) }
        }
    }
    return found
}

private func scanEntries(_ directory: URL, _ artifact: ModelArtifact, allowingExtraneous: Bool) throws {
    let paths = Set(artifact.files.map(\.path))
    let folders = Set(
        artifact.files.flatMap { file in
            let parts = file.path.split(separator: "/")
            return parts.indices.dropLast().map { parts[...$0].joined(separator: "/") }
        })
    let found = try entries(in: directory)
    for entry in found {
        if entry.mode == S_IFLNK { throw ManagedModelStoreError.symbolicLink }
        guard
            allowingExtraneous
                || (entry.mode == S_IFREG
                    ? paths.contains(entry.path) : entry.mode == S_IFDIR && folders.contains(entry.path))
        else { throw ManagedModelStoreError.unexpectedFile(entry.path) }
    }
    if !allowingExtraneous,
        let missing = paths.subtracting(Set(found.filter { $0.mode == S_IFREG }.map(\.path))).sorted().first
    {
        throw ManagedModelStoreError.missingFile(missing)
    }
}

private func verifyStructure(_ directory: URL, _ artifact: ModelArtifact, allowingExtraneous: Bool) throws {
    let manager = FileManager.default
    if itemType(directory) == .typeSymbolicLink { throw ManagedModelStoreError.symbolicLink }
    if itemType(directory) == .typeDirectory {
        try scanEntries(directory, artifact, allowingExtraneous: allowingExtraneous)
    }
    let files = artifact.files.sorted { $0.path < $1.path }
    for file in files {
        let attributes = try? manager.attributesOfItem(atPath: directory.appending(path: file.path).path)
        guard attributes?[.type] as? FileAttributeType == .typeRegular else {
            throw ManagedModelStoreError.missingFile(file.path)
        }
        guard (attributes?[.size] as? NSNumber)?.int64Value == file.bytes else {
            throw ManagedModelStoreError.sizeMismatch(file.path)
        }
    }
}

private func verifyFiles(_ directory: URL, _ artifact: ModelArtifact, allowingExtraneous: Bool = false) throws {
    try verifyStructure(directory, artifact, allowingExtraneous: allowingExtraneous)
    for file in artifact.files.sorted(by: { $0.path < $1.path }) {
        try Task.checkCancellation()
        guard try digest(directory.appending(path: file.path)) == file.sha256 else {
            throw ManagedModelStoreError.checksumMismatch(file.path)
        }
    }
}

private func verify(_ directory: URL, _ artifact: ModelArtifact) throws {
    try verifyFiles(directory, artifact)
    switch try family(artifact) {
    case .gliner:
        _ = try GLiNERAsset(directory: directory)
        let source = try JSONDecoder().decode(
            ContractSource.self, from: Data(contentsOf: directory.appending(path: "contract.json")))
        guard let upstream = artifact.upstream, source.model == upstream.repository,
            source.revision == upstream.revision
        else { throw ManagedModelStoreError.contractMismatch("upstream") }
    case .embedding(let precision):
        let bundle = try EmbeddingBundle(directory: directory)
        guard bundle.precision == precision else { throw ManagedModelStoreError.contractMismatch("precision") }
        guard bundle.revision == artifact.upstream?.revision else {
            throw ManagedModelStoreError.contractMismatch("upstream")
        }
    case .speech:
        try IrodoriBundleInventory.validateManifest(Data(contentsOf: directory.appending(path: "irodori.json")))
    case .image:
        let raw = try Data(contentsOf: directory.appending(path: "bundle.json"))
        try verifyImageManifest(raw, artifact)
        let manifest = try ImageGenerationManifest.decode(raw)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        for (kind, specification) in manifest.profile.components {
            let asset = directory.appending(path: specification.asset)
            let path = asset.deletingLastPathComponent().appending(path: "component.json")
            let component = try decoder.decode(ImageBundleComponent.self, from: Data(contentsOf: path))
            guard component.format == 2, component.kind == kind, component.asset == asset.lastPathComponent,
                component.conversionComplete, component.stateNames.isEmpty,
                component.profile == manifest.profile, component.inputs == specification.inputs,
                component.outputs == specification.outputs
            else { throw ManagedModelStoreError.contractMismatch("image component " + kind) }
        }
    }
}

private func verifyImageManifest(_ raw: Data, _ artifact: ModelArtifact) throws {
    let manifest = try ImageGenerationManifest.decode(raw)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let acceptance = try decoder.decode(ImageBundleAcceptance.self, from: raw)
    guard acceptance.verified, acceptance.numericVerified, acceptance.qualityVerified,
        manifest.profile.id == artifact.localBundle?.profileID,
        manifest.profile.source.model == artifact.upstream?.repository,
        manifest.profile.source.revision == artifact.upstream?.revision
    else { throw ManagedModelStoreError.contractMismatch("image profile") }
    let expected = artifact.files.filter { $0.path != "bundle.json" }
    guard manifest.assets.count == expected.count,
        expected.allSatisfy({ file in
            manifest.assets[file.path]?.bytes == file.bytes && manifest.assets[file.path]?.sha256 == file.sha256
        })
    else { throw ManagedModelStoreError.contractMismatch("image inventory") }
}

private struct ImageBundleAcceptance: Decodable {
    let verified: Bool
    let numericVerified: Bool
    let qualityVerified: Bool
}

private struct ImageBundleComponent: Decodable {
    let format: Int
    let kind: String
    let asset: String
    let conversionComplete: Bool
    let stateNames: [String]
    let profile: ImageGenerationProfile
    let inputs: [ImageTensorContract]
    let outputs: [ImageTensorContract]
}

private func verifyLocalManifest(_ raw: Data, _ artifact: ModelArtifact) throws {
    switch try family(artifact) {
    case .image: try verifyImageManifest(raw, artifact)
    case .speech: try IrodoriBundleInventory.validateManifest(raw)
    default: throw ManagedModelStoreError.invalidDefinition
    }
}
