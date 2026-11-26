struct OutputFileRecord: Encodable, Equatable {
    let function: String
    let name: String
    let rows: Int
    let width: Int
    let sha256: String
}
