public enum ChronosHistoryWindow {
    public static func make(history: [Float], truth: [Float], revealed: Int, maxLength: Int) -> [Float]? {
        guard !history.isEmpty, maxLength > 0, revealed >= 0, revealed <= truth.count,
            !history.contains(where: \.isInfinite), !truth.prefix(revealed).contains(where: \.isInfinite)
        else { return nil }
        let values = Array((history + truth.prefix(revealed)).suffix(maxLength))
        return values.contains(where: \.isFinite) ? values : nil
    }
}
