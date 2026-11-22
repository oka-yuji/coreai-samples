protocol LogitsSource {
    func reset()
    func prefill(_ input: [Int32], usePrefillFunction: Bool) async throws
    func step(_ token: Int32) async throws
    func logits() -> [Float16]
}
