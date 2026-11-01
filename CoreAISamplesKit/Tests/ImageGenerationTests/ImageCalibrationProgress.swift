#if os(iOS)
    import ImageGenerationCore
    import UIKit

    @MainActor
    final class ImageCalibrationProgress {
        private let window: UIWindow
        private weak var previous: UIWindow?
        private let count = UILabel()
        private let phase = UILabel()

        init() throws {
            guard
                let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene })
                    .first(where: {
                        $0.activationState == .foregroundActive && $0.windows.contains(where: \.isKeyWindow)
                    })
            else { throw ImageExecutionError.permissionDenied }
            previous = scene.windows.first(where: \.isKeyWindow)
            window = UIWindow(windowScene: scene)
            let controller = UIViewController()
            controller.view.backgroundColor = .systemBackground
            let title = UILabel()
            title.text = "Qwen Image verification"
            title.font = .preferredFont(forTextStyle: .title1)
            count.font = .preferredFont(forTextStyle: .title2)
            phase.font = .preferredFont(forTextStyle: .body)
            let help = UILabel()
            help.text = "Each image takes several minutes. Keep this app open until verification finishes."
            help.font = .preferredFont(forTextStyle: .footnote)
            help.textColor = .secondaryLabel
            let spinner = UIActivityIndicatorView(style: .large)
            spinner.startAnimating()
            for label in [title, count, phase, help] {
                label.numberOfLines = 0
                label.textAlignment = .center
                label.adjustsFontForContentSizeCategory = true
            }
            let stack = UIStackView(arrangedSubviews: [title, count, spinner, phase, help])
            stack.axis = .vertical
            stack.spacing = 24
            stack.translatesAutoresizingMaskIntoConstraints = false
            controller.view.addSubview(stack)
            NSLayoutConstraint.activate([
                stack.leadingAnchor.constraint(
                    equalTo: controller.view.safeAreaLayoutGuide.leadingAnchor, constant: 28),
                stack.trailingAnchor.constraint(
                    equalTo: controller.view.safeAreaLayoutGuide.trailingAnchor, constant: -28),
                stack.centerYAnchor.constraint(equalTo: controller.view.safeAreaLayoutGuide.centerYAnchor),
            ])
            window.rootViewController = controller
            window.windowLevel = .normal + 1
            window.makeKeyAndVisible()
        }

        func begin(_ number: Int, total: Int) {
            count.text = "Image \(number) of \(total)"
            phase.text = "Checking the model and inputs"
        }

        func update(_ value: ImageExecutionPhase) {
            switch value {
            case .preparing: phase.text = "Checking model files"
            case .loading: phase.text = "Preparing the next stage"
            case .running(.denoiser, let step): phase.text = "Generating image — step \((step ?? 0) + 1) of 8"
            case .running(.vaeDecoder, _): phase.text = "Finishing the image"
            case .running: phase.text = "Encoding the prompt or reference image"
            case .releasing: phase.text = "Releasing model resources"
            }
        }

        func close() {
            window.isHidden = true
            previous?.makeKey()
        }
    }
#endif
