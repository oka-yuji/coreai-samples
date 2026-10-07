import Foundation

extension DecisionResult {
    public func systemOneJSON(for request: DecisionRequest) throws -> Data {
        guard answers.count == request.questions.count else {
            throw DecisionError.invalidRequest("The result does not match the request.")
        }
        let quote = DecisionRenderer.jsonString
        func rounded(_ value: Float) -> String {
            String((Double(value) * 10_000).rounded(.toNearestOrEven) / 10_000)
        }
        let fields = try zip(answers, request.questions).map { answer, question in
            guard answer.id.utf8.elementsEqual(question.id.utf8) else {
                throw DecisionError.invalidRequest("The result question IDs do not match the request.")
            }
            let distribution = answer.distribution
            let probabilities =
                "{"
                + distribution.map { quote($0.option) + ":" + rounded($0.probability) }
                .joined(separator: ",") + "}"
            let content: String
            switch answer {
            case .choice(_, _, let top):
                content =
                    "\"type\":\"choice\",\"choice\":" + quote(top) + ",\"confidence\":"
                    + rounded(answer.maximumProbability) + ",\"probabilities\":" + probabilities
            case .noul(_, _, let probability):
                content = "\"type\":\"noul\",\"noul\":" + rounded(probability)
            case .score(_, _, let expected):
                guard case .score(let levels) = question.criteria else {
                    throw DecisionError.invalidRequest("The result question types do not match the request.")
                }
                let legend =
                    "{"
                    + levels.enumerated().map {
                        quote(String($0.offset)) + ":" + ($0.element.map(quote) ?? "null")
                    }.joined(separator: ",") + "}"
                content =
                    "\"type\":\"score\",\"score\":" + rounded(expected) + ",\"confidence\":"
                    + rounded(answer.maximumProbability) + ",\"legend\":" + legend + ",\"probabilities\":"
                    + probabilities
            }
            return quote(answer.id) + ":{" + content + "}"
        }
        let diagnostics = answers.map { answer in
            let values = answer.distribution.map {
                "{\"option\":" + quote($0.option) + ",\"probability\":" + String($0.probability)
                    + ",\"logit\":" + String($0.logit) + "}"
            }.joined(separator: ",")
            return "{\"id\":" + quote(answer.id) + ",\"top\":" + quote(answer.top) + ",\"distribution\":[" + values
                + "]}"
        }.joined(separator: ",")
        let json =
            "{\"model\":" + quote(model) + ",\"answers\":{" + fields.joined(separator: ",")
            + "},\"usage\":{\"input_tokens\":" + String(inputTokens) + ",\"output_tokens\":0},\"metadata\":{"
            + "\"model_version\":" + quote(modelVersion) + ",\"elapsed_seconds\":" + String(elapsedSeconds)
            + ",\"calibrated\":false,\"diagnostics\":[" + diagnostics + "]}}"
        return Data(json.utf8)
    }
}
