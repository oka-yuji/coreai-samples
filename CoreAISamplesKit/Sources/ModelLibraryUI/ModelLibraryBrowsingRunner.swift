import Foundation
import ModelLibraryFeature

public enum ModelLibraryBrowsingRunner {
    public static func run(_ effect: ModelLibraryEffect, send: (ModelLibraryAction) -> Void) {
        switch effect {
        case .inspect(_, let artifact, let request):
            send(.inspected(artifact.id, request, nil))
        case .download(_, let artifact, let request):
            send(.failed(artifact.id, request, .downloadNotConfigured))
        case .validate(_, let artifact, let request, _):
            send(.failed(artifact.id, request, .invalidResource))
        case .delete(let resource, let request):
            send(.failed(resource.artifactID, request, .unavailable))
        case .stopTransfer(let artifactID, let request):
            send(.transferStopped(artifactID, request))
        case .stopDemo(let session, let request):
            send(
                .demoStopFailed(
                    session.id, request,
                    String(localized: "Demo execution is not connected in this build.", bundle: .module)))
        case .prepareDemo(let session):
            send(.demoPreparationChanged(session.id, .unavailable))
        }
    }
}
