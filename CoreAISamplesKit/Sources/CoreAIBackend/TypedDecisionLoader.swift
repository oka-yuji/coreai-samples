import Foundation

actor TypedDecisionLoader<Value: Sendable> {
    typealias Loaded = (value: Value, bookmark: Data)

    private let bundle: TypedDecisionBundle
    private var bookmark: Data
    private let resolve: @Sendable (Data) throws -> Loaded?
    private let fresh: @Sendable (TypedDecisionBundle) async throws -> Loaded

    init(
        bundle: TypedDecisionBundle, bookmark: Data,
        resolve: @escaping @Sendable (Data) throws -> Loaded?,
        fresh: @escaping @Sendable (TypedDecisionBundle) async throws -> Loaded
    ) {
        self.bundle = bundle
        self.bookmark = bookmark
        self.resolve = resolve
        self.fresh = fresh
    }

    func load() async throws -> Value {
        try Task.checkCancellation()
        let loaded: Loaded
        if let cached = try resolve(bookmark) {
            loaded = cached
        } else {
            let verified = try TypedDecisionRuntime.revalidated(bundle)
            try Task.checkCancellation()
            loaded = try await fresh(verified)
        }
        try Task.checkCancellation()
        bookmark = loaded.bookmark
        return loaded.value
    }
}
