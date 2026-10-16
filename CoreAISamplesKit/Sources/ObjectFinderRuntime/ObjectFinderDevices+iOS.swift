#if os(iOS)
    import AVFoundation
    import Foundation
    import ObjectFinderFeature
    import UIKit

    extension LiveObjectFinderService {
        public convenience init() {
            let capture = ObjectFinderCapture()
            let finder = ObjectRegionFinder()
            self.init(
                devices: ObjectFinderDevices(
                    authorize: Self.authorizeCamera,
                    prepare: { try await finder.prepare(folder: $0, query: $1) },
                    startCamera: { try await capture.start(rotation: $0) },
                    stopCamera: { await capture.stop() },
                    checkCamera: { try await capture.checkActive() },
                    candidates: { try await finder.candidates(in: $0) },
                    closeFinder: { try await finder.close() },
                    beginOrientation: { UIDevice.current.beginGeneratingDeviceOrientationNotifications() },
                    endOrientation: { UIDevice.current.endGeneratingDeviceOrientationNotifications() },
                    notify: { UINotificationFeedbackGenerator().notificationOccurred(.success) }))
        }

        private static func authorizeCamera() async throws {
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: return
            case .notDetermined:
                guard await AVCaptureDevice.requestAccess(for: .video) else { throw ObjectFinderFailure.cameraDenied }
            case .denied, .restricted:
                throw ObjectFinderFailure.cameraDenied
            @unknown default:
                throw ObjectFinderFailure.cameraUnavailable
            }
        }
    }
#endif
