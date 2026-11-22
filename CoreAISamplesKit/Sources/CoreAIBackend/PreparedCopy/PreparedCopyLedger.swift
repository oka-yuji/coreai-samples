struct PreparedCopyLedger: Codable, Equatable, Sendable {
    var namespace: String
    var osBuild: String
    var assetHash: String
    var records: [String: PreparedCopyRecord]
}
