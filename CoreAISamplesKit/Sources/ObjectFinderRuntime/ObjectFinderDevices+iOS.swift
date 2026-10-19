#if os(iOS)
    import AVFoundation
    import Darwin
    import Foundation
    import ObjectFinderFeature
    import UIKit
    import os

    extension LiveObjectFinderService {
        public convenience init(preview: ObjectFinderPreview) {
            let capture = ObjectFinderCapture(preview: preview)
            let finder = ObjectRegionFinder()
            self.init(
                devices: ObjectFinderDevices(
                    authorize: Self.authorizeCamera,
                    prepare: { try await finder.prepare(folder: $0, query: $1) },
                    startCamera: {
                        let frames = try await capture.start(rotation: $0)
                        preview.connected()
                        return frames
                    },
                    stopCamera: { await capture.stop() },
                    checkCamera: { try await capture.checkActive() },
                    candidates: { try await finder.candidates(in: $0, mode: $1) },
                    closeFinder: { try await finder.close() },
                    beginOrientation: { UIDevice.current.beginGeneratingDeviceOrientationNotifications() },
                    endOrientation: { UIDevice.current.endGeneratingDeviceOrientationNotifications() },
                    notify: { UINotificationFeedbackGenerator().notificationOccurred(.success) },
                    requestFrame: { try await capture.requestFrame() },
                    trackCandidates: { tryFrame, candidates in
                        await capture.trackCandidates(candidates, frame: tryFrame)
                    },
                    memory: Self.sampleMemory,
                    trackedCandidates: { await capture.trackedCandidates() },
                    rescore: { try await finder.rescore($1, in: $0) }))
        }

        private static func sampleMemory() -> ObjectFinderMemory? {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let result = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                }
            }
            guard result == KERN_SUCCESS else { return nil }
            return ObjectFinderMemory(
                footprintBytes: info.phys_footprint, availableBytes: UInt64(os_proc_available_memory()))
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
