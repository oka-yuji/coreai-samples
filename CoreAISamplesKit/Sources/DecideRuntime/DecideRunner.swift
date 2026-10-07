import DecideFeature
import Foundation

@MainActor
public protocol DecideService {
    func examples() async throws -> [DecideExample]
    func load(path: String) async throws -> DecideBundleInfo
    func count(_ input: DecideInput) async throws -> Int
    func decide(_ input: DecideInput, reuse: Bool) async throws -> DecideOutput
    func read(_ url: URL) async throws -> String
    func waitForIdle() async
}

@MainActor
public final class DecideRunner {
    private let service: any DecideService

    public init(service: any DecideService) {
        self.service = service
    }

    public func run(_ effect: DecideEffect, send: (DecideAction) -> Void) async {
        switch effect {
        case .examples(let session):
            do { send(.examplesLoaded(session, try await service.examples())) } catch is CancellationError {} catch {
                send(.examplesFailed(session, error.localizedDescription))
            }
        case .load(let session, let path):
            do { send(.loaded(session, try await service.load(path: path))) } catch is CancellationError {} catch {
                send(.failed(session, error.localizedDescription))
            }
        case .count(let session, let revision, let input):
            do { send(.counted(session, revision, try await service.count(input))) } catch is CancellationError {} catch
            { send(.countFailed(session, revision, error.localizedDescription)) }
        case .read(let session, let url):
            do { send(.documentRead(session, try await service.read(url))) } catch is CancellationError {} catch {
                send(.failed(session, error.localizedDescription))
            }
        case .decide(let session, let revision, let input, let reuse):
            do {
                send(.completed(session, revision, try await service.decide(input, reuse: reuse)))
            } catch is CancellationError {} catch { send(.failed(session, error.localizedDescription)) }
        case .waitForIdle(let session):
            await service.waitForIdle()
            send(.stopped(session))
        case .export(let ticket, let output, let confirmations):
            do {
                send(.exportPrepared(ticket, try DecideExport.data(output: output, confirmations: confirmations)))
            } catch { send(.exportFailed(ticket, error.localizedDescription)) }
        }
    }
}
