public enum VoxCPMText {
    public static let whitespace: Set<Unicode.Scalar> = Set(
        [
            0x09...0x0D, 0x1C...0x20, 0x85...0x85, 0xA0...0xA0, 0x1680...0x1680, 0x2000...0x200A, 0x2028...0x2029,
            0x202F...0x202F, 0x205F...0x205F, 0x3000...0x3000,
        ]
        .flatMap { $0 }.compactMap(Unicode.Scalar.init))

    public static func normalized(_ text: String) throws -> String {
        guard text.unicodeScalars.contains(where: { !whitespace.contains($0) }) else { throw VoxCPMError.emptyText }
        var result = String.UnicodeScalarView()
        var inRun = false
        for scalar in text.unicodeScalars {
            if !whitespace.contains(scalar) {
                result.append(scalar)
                inRun = false
            } else if !inRun {
                result.append(" ")
                inRun = true
            }
        }
        return String(result)
    }
}
