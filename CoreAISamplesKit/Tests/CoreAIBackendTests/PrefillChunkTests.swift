import Foundation
import Testing

@testable import CoreAIBackend

@Suite struct PrefillChunkTests {
    typealias Asset = (modelID: String, mainHash: String)

    static let qwen08: Asset = ("Qwen/Qwen3.5-0.8B", "41b46bd34a69c7a25e017122323e72c338ff1fb1a78f17888e5e04bb0c77a2ba")
    static let qwen9: Asset = ("Qwen/Qwen3.5-9B", "410d97a4b3c16ac7b52d30de13ff43234957e080a27db08af67c9052241ee01e")
    static let qwen27: Asset = ("Qwen/Qwen3.8-27B", "ffa4677307b07d8d22073c9f5d8393574327e6b0cc07bd8ff4656f2a22dd1636")
    static let qwen06: Asset = ("Qwen/Qwen3-0.6B", "0f789ad2864edd643a71503da20e26ed75274a15597a2b5cd55fe7fe267d23a5")
    static let adopted = [qwen08, qwen9]
    static let memoryRule: [(physicalMemory: UInt64, chunk: Int)] = [
        (16 << 30, 2048), ((64 << 30) - 1, 2048), (64 << 30, 4096), (128 << 30, 4096),
    ]
    static let hashBytes = Data((0..<32).map { UInt8($0) })
    static let hashHex = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
    static let source08 = #"{"model_definition": "torch", "hf_model_id": "Qwen/Qwen3.5-0.8B"}"#

    static func bundle(name: String, source: String?, hash: Data?) throws -> URL {
        let field = source.map { #", "source": \#($0)"# } ?? ""
        let folder = try tinyBundle(
            metadata: #"{"name": "\#(name)", "assets": {"main": "tiny.aimodel"}, "#
                + #""language": {"vocab_size": 8, "max_context_length": 64}\#(field)}"#)
        try hash?.write(to: folder.appending(path: "tiny.aimodel/main.hash"))
        return folder
    }

    @Test(
        "An adopted hybrid asset uses a prefill chunk of 256 at every physical memory size",
        arguments: memoryRule.map(\.physicalMemory))
    func adoptedHybridAssetUses256(physicalMemory: UInt64) {
        for asset in Self.adopted {
            #expect(
                Engine.prefillChunkSize(
                    modelID: asset.modelID, assetHash: asset.mainHash, isHybrid: true,
                    physicalMemory: physicalMemory, adopted: Self.adopted) == 256)
        }
    }

    @Test(
        "A non-hybrid, unlisted or unidentified asset keeps 4096 from 64 GiB of physical memory and 2048 below",
        arguments: [
            (qwen08.modelID, qwen08.mainHash, false), (qwen06.modelID, qwen06.mainHash, false),
            (qwen08.modelID, "00", true), (qwen08.modelID, qwen9.mainHash, true), (qwen08.modelID, nil, true),
            (nil, qwen08.mainHash, true), (nil, nil, true), ("qwen/qwen3.5-0.8b", qwen08.mainHash, true),
            (qwen27.modelID, qwen27.mainHash, true),
        ] as [(String?, String?, Bool)])
    func otherAssetsKeepTheMemoryRule(modelID: String?, assetHash: String?, isHybrid: Bool) {
        for (physicalMemory, chunk) in Self.memoryRule {
            #expect(
                Engine.prefillChunkSize(
                    modelID: modelID, assetHash: assetHash, isHybrid: isHybrid, physicalMemory: physicalMemory,
                    adopted: Self.adopted) == chunk)
        }
    }

    @Test("The shipped adoption table never lists Qwen3.8-27B, so 27B keeps the physical-memory chunk")
    func shippedTableKeeps27BOnTheMemoryRule() {
        #expect(!Engine.chunk256Assets.contains { $0.modelID == Self.qwen27.modelID })
        for (physicalMemory, chunk) in Self.memoryRule {
            #expect(
                Engine.prefillChunkSize(
                    modelID: Self.qwen27.modelID, assetHash: Self.qwen27.mainHash, isHybrid: true,
                    physicalMemory: physicalMemory) == chunk)
        }
    }

    @Test(
        "A renamed bundle keeps the adopted chunk through its source model ID and main.hash, and the name alone selects nothing",
        arguments: [
            ("qwen3_5_0_8b_qwen3_5_cfg_dynamic", source08, 256), ("renamed-copy", source08, 256),
            ("qwen3_5_0_8b_qwen3_5_cfg_dynamic", nil, 4096),
        ] as [(String, String?, Int)])
    func renamedBundleKeepsTheAdoptedChunk(name: String, source: String?, chunk: Int) throws {
        let folder = try Self.bundle(name: name, source: source, hash: Self.hashBytes)
        defer { try? FileManager.default.removeItem(at: folder) }
        let bundle = try ModelBundle(url: folder)
        #expect(bundle.name == name)
        #expect(
            Engine.prefillChunkSize(
                modelID: bundle.sourceModelID, assetHash: bundle.mainHash, isHybrid: true,
                physicalMemory: 128 << 30, adopted: [(Self.qwen08.modelID, Self.hashHex)]) == chunk)
    }

    @Test("ModelBundle reads source.hf_model_id and the bytes of main.hash as lowercase hex")
    func bundleReadsTheSourceModelIDAndMainHash() throws {
        let folder = try Self.bundle(name: "tiny", source: Self.source08, hash: Self.hashBytes)
        defer { try? FileManager.default.removeItem(at: folder) }
        let bundle = try ModelBundle(url: folder)
        #expect(bundle.sourceModelID == "Qwen/Qwen3.5-0.8B")
        #expect(bundle.mainHash == Self.hashHex)
    }

    @Test(
        "A bundle without a readable source.hf_model_id or main.hash still opens and reports neither",
        arguments: [nil, "null", #"{"model_definition": "torch"}"#, #"{"hf_model_id": 5}"#, #""Qwen/Qwen3.5-0.8B""#]
            as [String?])
    func bundleWithoutIdentityStillOpens(source: String?) throws {
        let folder = try Self.bundle(name: "tiny", source: source, hash: nil)
        defer { try? FileManager.default.removeItem(at: folder) }
        let bundle = try ModelBundle(url: folder)
        #expect(bundle.sourceModelID == nil)
        #expect(bundle.mainHash == nil)
    }
}
