#if os(iOS)
    import AVFAudio
    import Foundation
    import OSLog
    import SpeechCore

    @MainActor
    final class PhonePlaybackSessionDevice: PlaybackSessionDevice {
        private let session = AVAudioSession.sharedInstance()
        private let center = NotificationCenter.default
        private var observers: [NSObjectProtocol] = []
        private let logger = Logger(subsystem: "com.coreai-samples.SpeechPhone", category: "Playback")

        func activate(
            interrupted: @escaping @MainActor () -> Void,
            ready: @escaping @MainActor (Result<Void, any Error>) -> Void
        ) {
            do { try session.setCategory(.playback, mode: .spokenAudio) } catch {
                ready(.failure(error))
                return
            }
            let names = [
                AVAudioSession.didBecomeInactiveNotification, AVAudioSession.routeChangeNotification,
                AVAudioSession.mediaServicesWereResetNotification,
            ]
            observers = names.map { name in
                center.addObserver(forName: name, object: nil, queue: .main) { notification in
                    let route = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt
                    let shouldStop =
                        notification.name == AVAudioSession.mediaServicesWereResetNotification
                        || notification.name == AVAudioSession.didBecomeInactiveNotification
                        || route == AVAudioSession.RouteChangeReason.oldDeviceUnavailable.rawValue
                    if shouldStop { DispatchQueue.main.async { interrupted() } }
                }
            }
            session.activate(options: []) { [weak self] activated, error in
                DispatchQueue.main.async {
                    if activated {
                        ready(.success(()))
                    } else {
                        self?.removeObservers()
                        ready(.failure(error ?? SpeechError.playbackFailed))
                    }
                }
            }
        }
        func deactivate(completed: @escaping @MainActor () -> Void) {
            removeObservers()
            session.deactivate(options: .notifyOthersOnDeactivation) { [weak self] deactivated, error in
                DispatchQueue.main.async {
                    if !deactivated {
                        self?.logger.warning(
                            "Audio session deactivation failed: \(error?.localizedDescription ?? "Unknown error")")
                    }
                    completed()
                }
            }
        }
        private func removeObservers() {
            for observer in observers { center.removeObserver(observer) }
            observers.removeAll()
        }
    }
#endif
