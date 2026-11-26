struct PositionLogitsRecord: Encodable {
    var record = "logits"
    let tokenCount: Int
    let tokensSha256: String
    let start: Int
    let chunk: Int
    let shape: [Int]
    var dtype = "float16"
    var byteOrder = "little"
    let prefixFunction: String?
    let prefixChunk: Int
    let prefixCalls: Int
    let mainCalls: Int
    let mainHash: String?
    let requestedCompute: String?
    let expectFrequentReshapes: Bool?
    let cached: Bool
    let cacheQueryError: String?
    let specializeS: Double
    let nonFiniteCount: Int
    let loadS: Double
    let prefixS: Double
    let scoringS: Double
    let scoringCallS: Double
    var outputs: [OutputFileRecord]?
}
