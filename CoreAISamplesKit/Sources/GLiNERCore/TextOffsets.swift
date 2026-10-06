extension String {
    /// The UTF-16 range (for `NSString` and `AttributedString`) of a range counted in Unicode scalars.
    public func utf16Range(fromScalarRange range: Range<Int>) -> Range<Int>? {
        guard let lower = utf16Offset(ofScalarOffset: range.lowerBound),
            let upper = utf16Offset(ofScalarOffset: range.upperBound)
        else { return nil }
        return lower..<upper
    }

    /// The scalar range of a UTF-16 range, or `nil` when a bound splits a surrogate pair.
    public func scalarRange(fromUTF16Range range: Range<Int>) -> Range<Int>? {
        guard let lower = scalarOffset(ofUTF16Offset: range.lowerBound),
            let upper = scalarOffset(ofUTF16Offset: range.upperBound)
        else { return nil }
        return lower..<upper
    }

    private func utf16Offset(ofScalarOffset offset: Int) -> Int? {
        guard offset >= 0,
            let index = unicodeScalars.index(
                unicodeScalars.startIndex, offsetBy: offset, limitedBy: unicodeScalars.endIndex)
        else { return nil }
        return utf16.distance(from: utf16.startIndex, to: index)
    }

    private func scalarOffset(ofUTF16Offset offset: Int) -> Int? {
        guard offset >= 0,
            let index = utf16.index(utf16.startIndex, offsetBy: offset, limitedBy: utf16.endIndex),
            let scalarIndex = index.samePosition(in: unicodeScalars)
        else { return nil }
        return unicodeScalars.distance(from: unicodeScalars.startIndex, to: scalarIndex)
    }
}
