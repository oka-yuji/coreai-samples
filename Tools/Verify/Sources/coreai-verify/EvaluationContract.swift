struct EvaluationContract: Decodable {
    struct Generation: Decodable {
        let eosTokenId: [Int32]
    }

    let version: String
    let roles: [String: String]
    let generation: Generation
}
