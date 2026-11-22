struct PreparedCopyKey: Codable, Equatable, Sendable {
    let namespace: String
    let osBuild: String
    let assetHash: String
    let options: String
}
