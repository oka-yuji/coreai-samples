import Foundation

private final class ImageTestBundle: NSObject {}

extension Bundle {
    static var module: Bundle { Bundle(for: ImageTestBundle.self) }
}
