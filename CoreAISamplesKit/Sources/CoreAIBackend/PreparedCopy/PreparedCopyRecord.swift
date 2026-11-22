struct PreparedCopyRecord: Codable, Equatable, Sendable {
    var marks: [PreparedCopyMark]
    var recoveries: Int
    var suspects: [PreparedCopyMark]
    var failed: Bool
}
