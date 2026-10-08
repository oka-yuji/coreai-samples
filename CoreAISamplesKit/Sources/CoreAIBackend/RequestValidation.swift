import FoundationModels

extension Runtime {
    static func validatedTools(for request: LanguageModelExecutorGenerationRequest) throws -> [ToolSchema] {
        if request.schema != nil {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription: "Guided generation is not supported. Request plain text instead."))
        }
        if !request.metadata.isEmpty {
            throw LanguageModelError.unsupportedGenerationGuide(
                .init(
                    schemaName: nil,
                    debugDescription:
                        "Custom request metadata is not supported: \(request.metadata.keys.sorted().joined(separator: ", ")). Use GenerationOptions or ContextOptions instead."
                ))
        }
        if let level = request.contextOptions.reasoningLevel {
            switch level {
            case .light, .moderate, .deep: break
            case .custom(let name):
                guard ["off", "low", "medium", "xhigh"].contains(name) else {
                    throw LanguageModelError.unsupportedCapability(
                        .init(
                            capability: .reasoning,
                            debugDescription:
                                "Reasoning level '\(name)' is not supported. Use off, low, medium or xhigh."))
                }
            @unknown default:
                throw LanguageModelError.unsupportedCapability(
                    .init(
                        capability: .reasoning,
                        debugDescription: "This reasoning level is not supported. Use a documented level."))
            }
        }
        if let mode = request.generationOptions.samplingMode?.kind {
            switch mode {
            case .greedy, .randomTopK, .randomProbabilityThreshold: break
            @unknown default:
                throw LanguageModelError.unsupportedGenerationGuide(
                    .init(
                        schemaName: nil,
                        debugDescription: "This sampling mode is not supported. Use greedy, top-k or top-p."))
            }
        }
        if let mode = request.generationOptions.toolCallingMode?.kind {
            switch mode {
            case .allowed, .disallowed, .required: break
            @unknown default:
                throw LanguageModelError.unsupportedCapability(
                    .init(
                        capability: .toolCalling,
                        debugDescription: "This tool-calling mode is not supported. Use allowed or disallowed."))
            }
        }
        if let prompt = request.transcript.reversed().compactMap({ entry -> Transcript.Prompt? in
            if case .prompt(let prompt) = entry { return prompt }
            return nil
        }).first, prompt.responseFormat != nil {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription: "The current prompt requests structured output. Request plain text instead."))
        }
        for entry in request.transcript {
            let segments: [Transcript.Segment]
            switch entry {
            case .instructions(let value): segments = value.segments
            case .prompt(let value): segments = value.segments
            case .response(let value): segments = value.segments
            case .reasoning(let value): segments = value.segments
            case .toolOutput(let value): segments = value.segments
            case .toolCalls: continue
            @unknown default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(
                        unsupportedContent: [entry],
                        debugDescription: "This transcript entry is not supported. Use text messages and tool calls."))
            }
            for segment in segments {
                guard case .text = segment else {
                    throw LanguageModelError.unsupportedTranscriptContent(
                        .init(
                            unsupportedContent: [entry],
                            debugDescription:
                                "Only text segments are supported. Convert structured content and attachments to text.")
                    )
                }
            }
        }
        return try tools(request)
    }
}
