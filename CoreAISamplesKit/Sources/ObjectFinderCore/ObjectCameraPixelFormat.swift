public enum ObjectCameraPixelFormat: Hashable, Sendable {
    case bgra, yuvVideoRange, yuvFullRange

    public static func select(native: Self?, supported: Set<Self>) -> Self? {
        if let native, supported.contains(native) { return native }
        return supported.contains(.bgra) ? .bgra : nil
    }
}
