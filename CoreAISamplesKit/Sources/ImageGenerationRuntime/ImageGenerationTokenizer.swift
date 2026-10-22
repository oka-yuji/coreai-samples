import Foundation
import Hub
import ImageGenerationCore
import Tokenizers

public struct ImageGenerationTokenizer: Sendable {
    private let tokenizer: PreTrainedTokenizer
    private let drop: Int

    public init(directory: URL) throws {
        tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: HubApi.shared.configuration(fileURL: directory.appending(path: "tokenizer_config.json")),
            tokenizerData: HubApi.shared.configuration(fileURL: directory.appending(path: "tokenizer.json")))
        let system = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        drop = tokenizer.encode(text: system, addSpecialTokens: false).count
        guard tokenizer.encode(text: "<|image_pad|>", addSpecialTokens: false) == [ImagePromptLayout.imageToken] else {
            throw ImageGenerationError.invalidProfile("image token")
        }
    }

    public func prepare(_ prompt: String, hasReference: Bool) throws -> ImagePromptLayout {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ImageGenerationError.invalidTokens
        }
        guard prompt.utf8.count <= 262_144 else { throw ImageGenerationError.inputTooLarge(prompt.utf8.count) }
        let prefix =
            hasReference
            ? "<image1><|vision_start|>" + String(repeating: "<|image_pad|>", count: 256) + "<|vision_end|>" : ""
        let template =
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n"
            + prefix + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        return try ImagePromptLayout(
            tokens: tokenizer.encode(text: template, addSpecialTokens: false), drop: drop, hasReference: hasReference)
    }
}
