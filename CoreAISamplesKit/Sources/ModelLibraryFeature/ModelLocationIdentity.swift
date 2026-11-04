import Foundation

public enum ModelLocationIdentity {
    public static func key(_ location: URL, legacy: URL, managed: URL) -> String {
        let path = normalized(location)
        return path == normalized(legacy) ? normalized(managed) : path
    }

    private static func normalized(_ location: URL) -> String {
        let path = location.standardizedFileURL.path(percentEncoded: false)
        return path.count > 1 && path.hasSuffix("/") ? String(path.dropLast()) : path
    }
}
