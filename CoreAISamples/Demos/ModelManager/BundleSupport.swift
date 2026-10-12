import CoreAIBackend
import Foundation

enum BundleSupport {
    static let supportedModelIDPrefixes = ["Qwen/Qwen3-", "Qwen/Qwen3.5-", "Qwen/Qwen3.8-"]

    struct Refusal: LocalizedError, Equatable {
        let reason: String

        var errorDescription: String? { reason }
    }

    static func supportedBundle(at url: URL) throws -> ModelBundle {
        let bundle = try ModelBundle(url: url)
        let metadata = try? JSONSerialization.jsonObject(with: Data(contentsOf: url.appending(path: "metadata.json")))
        let source = (metadata as? [String: Any])?["source"] as? [String: Any]
        guard let modelID = source?["hf_model_id"] as? String, !modelID.isEmpty else {
            throw Refusal(
                reason:
                    "metadata.json has no source.hf_model_id, so this sample cannot tell which model family the bundle is."
            )
        }
        guard supportedModelIDPrefixes.contains(where: modelID.hasPrefix) else {
            throw Refusal(
                reason: "This sample runs Qwen3, Qwen3.5 and Qwen3.8 bundles made with Apple's converter. "
                    + "\(bundle.name) was converted from \(modelID).")
        }
        return bundle
    }
}
