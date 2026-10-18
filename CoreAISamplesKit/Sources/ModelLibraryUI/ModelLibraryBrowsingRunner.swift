import Foundation
import ModelLibraryFeature

public enum ModelLibraryBrowsingRunner {
    public static func run(_ effect: ModelLibraryEffect, send: (ModelLibraryAction) -> Void) {
        switch effect {
        case .inspect(let model, let request):
            send(.inspected(model.id, request, nil))
        case .download(let model, let request):
            send(.failed(model.id, request, .downloadNotConfigured))
        case .validate(let model, let request, _):
            send(.failed(model.id, request, .invalidResource))
        case .delete(let resource, let request):
            send(.failed(resource.modelID, request, .unavailable))
        case .stopTransfer(let modelID, let request):
            send(.transferStopped(modelID, request))
        case .stopDemo(let session, let request):
            send(
                .demoStopFailed(
                    session.id, request,
                    String(localized: "Demo execution is not connected in this build.", bundle: .module)))
        }
    }
}
