import CoreGraphics
import Foundation
import FoundationModels
import Testing

@_spi(Verification) @testable import CoreAIBackend

@Suite struct RequestValidationTests {
    private func prompt(_ text: String, options: GenerationOptions = .init(), context: ContextOptions = .init())
        -> Transcript.Entry
    {
        .prompt(.init(segments: [.text(.init(content: text))], options: options, contextOptions: context))
    }

    private func request(
        entries: [Transcript.Entry]? = nil, schema: GenerationSchema? = nil, options: GenerationOptions = .init(),
        context: ContextOptions = .init(), metadata: [String: any ConvertibleToGeneratedContent] = [:]
    ) -> LanguageModelExecutorGenerationRequest {
        LanguageModelExecutorGenerationRequest(
            id: UUID(), transcript: Transcript(entries: entries ?? [prompt("Hello")]), enabledTools: [], schema: schema,
            generationOptions: options, contextOptions: context, metadata: metadata)
    }

    @Test("Request schemas and a current prompt response format are refused without a model")
    func rejectsStructuredOutput() throws {
        let schema = AddArguments.generationSchema
        let formatted = Transcript.Entry.prompt(
            .init(segments: [.text(.init(content: "Hello"))], responseFormat: .init(schema: schema)))
        for input in [request(schema: schema), request(entries: [formatted])] {
            do {
                _ = try Runtime.validatedTools(for: input)
                Issue.record("Structured output was accepted")
            } catch LanguageModelError.unsupportedCapability(let detail) {
                #expect(detail.capability == .guidedGeneration)
                #expect(detail.debugDescription.contains("plain text"))
            }
        }
        #expect(
            try Runtime.validatedTools(for: request(entries: [formatted, prompt("A new plain-text question")])).isEmpty)
    }

    @Test("Custom request metadata is refused with its key names and a supported alternative")
    func rejectsMetadata() throws {
        do {
            _ = try Runtime.validatedTools(for: request(metadata: ["passthrough": "private value"]))
            Issue.record("Custom metadata was accepted")
        } catch LanguageModelError.unsupportedGenerationGuide(let detail) {
            #expect(detail.debugDescription.contains("passthrough"))
            #expect(detail.debugDescription.contains("GenerationOptions"))
            #expect(!detail.debugDescription.contains("private value"))
        }
    }

    @Test("Unknown custom reasoning levels are refused with the requested value", arguments: ["unexpected", "", "LOW"])
    func rejectsUnknownReasoning(level: String) throws {
        do {
            _ = try Runtime.validatedTools(for: request(context: .init(reasoningLevel: .custom(level))))
            Issue.record("An unsupported reasoning level was accepted")
        } catch LanguageModelError.unsupportedCapability(let detail) {
            #expect(detail.capability == .reasoning)
            #expect(detail.debugDescription.contains("'\(level)'"))
        }
    }

    @Test("Structured and image transcript segments are refused before rendering")
    func rejectsNonText() throws {
        let provider = try #require(CGDataProvider(data: Data([0, 0, 0, 255]) as CFData))
        let image = try #require(
            CGImage(
                width: 1, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent))
        let segments: [Transcript.Segment] = [
            .structure(.init(schemaName: "value", content: try GeneratedContent(json: #"{"value":1}"#))),
            .attachment(.init(content: .image(.init(image)))),
        ]
        for segment in segments {
            let entry = Transcript.Entry.prompt(.init(segments: [segment]))
            do {
                _ = try Runtime.validatedTools(for: request(entries: [entry]))
                Issue.record("A non-text segment was accepted")
            } catch LanguageModelError.unsupportedTranscriptContent(let detail) {
                #expect(detail.unsupportedContent.map(\.id) == [entry.id])
                #expect(detail.debugDescription.contains("text"))
            }
        }
    }

    @Test("Supported options retain their sampling and template mappings")
    func acceptsSupportedOptions() throws {
        let modes: [GenerationOptions.SamplingMode?] = [
            nil, .greedy, .random(top: 40, seed: 7), .random(probabilityThreshold: 0.95, seed: 7),
        ]
        for mode in modes {
            let options = GenerationOptions(samplingMode: mode, temperature: 0.7, maximumResponseTokens: 16)
            #expect(try Runtime.validatedTools(for: request(options: options)).isEmpty)
        }
        for mode in [GenerationOptions.ToolCallingMode.allowed, .disallowed] {
            #expect(try Runtime.validatedTools(for: request(options: .init(toolCallingMode: mode))).isEmpty)
        }
        for level in [
            ContextOptions.ReasoningLevel.light, .moderate, .deep, .custom("off"), .custom("low"), .custom("medium"),
            .custom("xhigh"),
        ] {
            #expect(try Runtime.validatedTools(for: request(context: .init(reasoningLevel: level))).isEmpty)
        }
        #expect(Runtime.templateContext(.custom("off"))["enable_thinking"] as? Bool == false)
        #expect(Runtime.templateContext(.light)["reasoning_effort"] as? String == "low")
        #expect(Runtime.templateContext(.moderate)["reasoning_effort"] as? String == "medium")
        #expect(Runtime.templateContext(.deep)["reasoning_effort"] as? String == "xhigh")
        do {
            _ = try Runtime.validatedTools(for: request(options: .init(toolCallingMode: .required)))
            Issue.record("Required tool calling was accepted")
        } catch LanguageModelError.unsupportedCapability(let detail) {
            #expect(detail.debugDescription.contains("force a tool call"))
        }
    }

    @Test(
        "Historical prompt settings do not override the current request and schema hints without a schema have no effect"
    )
    func preservesHistoricalSettings() throws {
        let old = prompt(
            "First", options: .init(samplingMode: .greedy, maximumResponseTokens: 2),
            context: .init(reasoningLevel: .custom("off")))
        let current = prompt("Second", options: .init(maximumResponseTokens: 3), context: .init(reasoningLevel: .light))
        for include in [true, false] {
            let input = request(
                entries: [old, current], options: .init(maximumResponseTokens: 3),
                context: .init(includeSchemaInPrompt: include, reasoningLevel: .light))
            #expect(try Runtime.validatedTools(for: input).isEmpty)
        }
    }

    @Test(
        "Real bundle templates accept the documented reasoning values and unsupported values fail before rendering",
        .enabled(if: !checkPaths.isEmpty), arguments: checkPaths)
    func checksRealTemplates(path: String) async throws {
        let bundle = try ModelBundle(url: URL(filePath: path))
        let tokenizer = try await bundle.loadTokenizer()
        for name in ["off", "low", "medium", "xhigh"] {
            let input = request(context: .init(reasoningLevel: .custom(name)))
            _ = try Runtime.validatedTools(for: input)
            let ids = try tokenizer.applyChatTemplate(
                messages: Runtime.messages(input.transcript), tools: nil,
                additionalContext: Runtime.templateContext(.custom(name)))
            #expect(!ids.isEmpty)
        }
        do {
            _ = try Runtime.validatedTools(for: request(context: .init(reasoningLevel: .custom("unsupported"))))
            Issue.record("An unsupported template value was accepted")
        } catch LanguageModelError.unsupportedCapability(let detail) {
            #expect(detail.debugDescription.contains("unsupported"))
        }
    }
}

extension ModelIntegrationTests {
    @Test(
        "Rejected reasoning options leave the live engine state unchanged", .enabled(if: !bundlePaths.isEmpty),
        arguments: bundlePaths)
    func rejectedRequestPreservesState(path: String) async throws {
        let model = try await LocalLanguageModel(bundle: URL(filePath: path))
        let session = LanguageModelSession(model: model)
        _ = try await session.respond(to: "Hello", options: .init(maximumResponseTokens: 1))
        let before = await model.processedTokens()
        do {
            _ = try await session.respond(
                to: "Another turn", contextOptions: .init(reasoningLevel: .custom("unsupported")))
            Issue.record("The live runtime accepted an unsupported reasoning level")
        } catch LanguageModelError.unsupportedCapability(let detail) {
            #expect(detail.debugDescription.contains("unsupported"))
        }
        #expect(await model.processedTokens() == before)
    }
}
