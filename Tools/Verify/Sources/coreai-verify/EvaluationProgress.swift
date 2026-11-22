struct EvaluationProgress: Encodable {
    var record = "evaluate"
    let id: String
    let inputTokens: Int
    let outputTokens: Int
    let stop: String
    let seconds: Double
}
