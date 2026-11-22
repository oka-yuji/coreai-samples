struct TopTwo: Encodable, Equatable {
    let first: Int32
    let firstProbability: Double
    let second: Int32
    let secondProbability: Double

    func encode(to encoder: any Encoder) throws {
        var entry = encoder.unkeyedContainer()
        try entry.encode(first)
        try entry.encode(firstProbability)
        try entry.encode(second)
        try entry.encode(secondProbability)
    }
}
