import Foundation
import GLiNERRuntime
import Testing

/// `contract.json` of the GL4 FP32 asset (`gliner25_multi_fp32_t256_t512`) without `model` and `revision`.
private let contract = #"""
    {
      "functions": {
        "encode_t256": {
          "inputs": ["input_ids", "attention_mask", "word_idx", "word_mask", "query_idx", "query_mask", "cls_idx",
                     "cls_mask"],
          "outputs": ["start_logits", "end_logits", "inside_prefix", "inside_mean", "null_logits", "cls_logits",
                      "pool_start", "pool_end", "scorer_start", "scorer_end", "content_prefix", "query_proj"]
        },
        "pair_t256": {
          "inputs": ["scorer_start", "scorer_end", "content_prefix", "query_proj", "start_logits", "end_logits",
                     "inside_prefix", "inside_mean", "word_mask", "query_mask", "cand_start", "cand_end", "cand_mask",
                     "cand_prior"],
          "outputs": ["pair_logits"]
        },
        "encode_t512": {
          "inputs": ["input_ids", "attention_mask", "word_idx", "word_mask", "query_idx", "query_mask", "cls_idx",
                     "cls_mask"],
          "outputs": ["start_logits", "end_logits", "inside_prefix", "inside_mean", "null_logits", "cls_logits",
                      "pool_start", "pool_end", "scorer_start", "scorer_end", "content_prefix", "query_proj"]
        },
        "pair_t512": {
          "inputs": ["scorer_start", "scorer_end", "content_prefix", "query_proj", "start_logits", "end_logits",
                     "inside_prefix", "inside_mean", "word_mask", "query_mask", "cand_start", "cand_end", "cand_mask",
                     "cand_prior"],
          "outputs": ["pair_logits"]
        }
      },
      "caps": {"tokens": [256, 512], "queries": 16, "class_labels": 16, "candidates": 192, "boundaries": "T + 1",
               "words": "T"},
      "host": {"pool_boundary_top_k": 32, "pool_size": 192, "min_pool_per_query": 8, "mask_logit": -10000.0,
               "quota_score_base": 5000.0, "pair_temperature": 1.0, "classification_temperature": 1.0,
               "abstention_threshold": 0.5, "overlap_policy": "flat", "default_threshold": 0.5,
               "token_pooling": "first", "append_period_unless_text_ends_with": [".", "!", "?"]}
    }
    """#

/// A temporary asset folder with the contract after `edit` and `models` empty `.aimodel` folders.
private func assetFolder(models: Int = 1, _ edit: (inout [String: [String: Any]]) -> Void = { _ in }) throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appending(path: "gliner-asset-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    var json = try #require(JSONSerialization.jsonObject(with: Data(contract.utf8)) as? [String: [String: Any]])
    edit(&json)
    try JSONSerialization.data(withJSONObject: json).write(to: folder.appending(path: "contract.json"))
    for index in 0..<models {
        try FileManager.default.createDirectory(
            at: folder.appending(path: "model\(index).aimodel"), withIntermediateDirectories: true)
    }
    return folder
}

struct GLiNERAssetTests {
    @Test("The asset accepts the GL4 contract and finds its one .aimodel")
    func acceptsTheGL4Contract() throws {
        let folder = try assetFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(try GLiNERAsset(directory: folder).modelURL.lastPathComponent == "model0.aimodel")
    }

    @Test("A contract that differs from the implementation, or a folder without exactly one .aimodel, is refused")
    func refusesOtherContracts() throws {
        let folders = try [
            assetFolder { $0["functions"]?["pair_t512"] = nil },
            assetFolder { $0["caps"]?["tokens"] = [256] },
            assetFolder { $0["host"]?["pool_size"] = 128 },
            assetFolder(models: 2),
            assetFolder(models: 0),
        ]
        defer { for folder in folders { try? FileManager.default.removeItem(at: folder) } }
        #expect(throws: GLiNERAssetError.contractMismatch("functions")) { try GLiNERAsset(directory: folders[0]) }
        #expect(throws: GLiNERAssetError.contractMismatch("caps")) { try GLiNERAsset(directory: folders[1]) }
        #expect(throws: GLiNERAssetError.contractMismatch("host")) { try GLiNERAsset(directory: folders[2]) }
        #expect(throws: GLiNERAssetError.modelCount(2)) { try GLiNERAsset(directory: folders[3]) }
        #expect(throws: GLiNERAssetError.modelCount(0)) { try GLiNERAsset(directory: folders[4]) }
    }
}
