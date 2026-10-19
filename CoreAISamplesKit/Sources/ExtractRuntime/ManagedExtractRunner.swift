import ExtractFeature
import Foundation
import GLiNERCore

@MainActor
public protocol ExtractServicing {
    func open(folder: URL) async throws -> AssetInfo

    func extract(text: String, schema: FormSchema, filter: CandidateFilter) async throws -> (
        mapping: FormMapping, timing: ExtractTiming
    )

    func release() async throws
}

@MainActor
public final class ManagedExtractRunner {
    static let unavailable = "The model is not available for this demo right now."

    private let service: any ExtractServicing
    private let isPermitted: @MainActor () -> Bool
    private let describe: @MainActor (any Error) -> String
    private var operations = 0
    private var waiting: [CheckedContinuation<Void, Never>] = []

    public init(
        service: any ExtractServicing, isPermitted: @escaping @MainActor () -> Bool,
        describe: @escaping @MainActor (any Error) -> String
    ) {
        self.service = service
        self.isPermitted = isPermitted
        self.describe = describe
    }

    public func run(_ effect: ExtractEffect, send: (ExtractAction) -> Void) async {
        switch effect {
        case .openAsset(let path):
            guard isPermitted() else { return send(.assetFailed(path: path, message: Self.unavailable)) }
            operations += 1
            defer { finish() }
            do {
                var info = try await service.open(folder: URL(filePath: path, directoryHint: .isDirectory))
                info.path = path
                send(.assetOpened(info))
            } catch is CancellationError {
                return
            } catch {
                send(.assetFailed(path: path, message: describe(error)))
            }
        case .extract(let version, let text, let form, let filter):
            guard isPermitted() else { return send(.extractionFailed(version: version, message: Self.unavailable)) }
            operations += 1
            defer { finish() }
            do {
                let (mapping, timing) = try await service.extract(text: text, schema: form, filter: filter)
                send(.extractionFinished(version: version, mapping: mapping, timing: timing))
            } catch is CancellationError {
                return
            } catch {
                send(.extractionFailed(version: version, message: describe(error)))
            }
        case .release:
            do {
                try await release()
                send(.released)
            } catch {
                send(.releaseFailed(describe(error)))
            }
        case .settle, .export:
            return
        }
    }

    public func release() async throws {
        while operations > 0 { await withCheckedContinuation { waiting.append($0) } }
        try await service.release()
    }

    private func finish() {
        operations -= 1
        guard operations == 0 else { return }
        let resumed = waiting
        waiting = []
        for continuation in resumed { continuation.resume() }
    }
}
