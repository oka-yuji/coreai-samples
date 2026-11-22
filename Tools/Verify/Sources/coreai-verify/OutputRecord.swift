struct OutputRecord: Encodable, Equatable {
    let id: String
    let inputSha256: String
    let inputTokens: Int
    var outputIds: [Int32] = []
    var stop = "limit"
    var stopToken: Int32?
    var top2: [TopTwo] = []
    var error: String?
    var nonFinite = 0
}
