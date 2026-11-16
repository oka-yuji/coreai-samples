import Foundation
import SpeechCore
import Testing
import VoxCPMTokenizers

@testable import SpeechAssets

struct VoxCPMManifestTests {
    @Test("VIN-03 Each bundled VoxCPM2 manifest passes the check and gives its configuration and contract values")
    func bundledManifests() throws {
        for profile in VoxCPMBundleProfile.allCases {
            let data = try VoxCPMBundleFixture.manifest(profile)
            #expect(try VoxCPMBundleInventory.profile(in: data) == profile)
            let manifest = try VoxCPMBundleInventory.manifest(in: data)
            #expect(manifest.configuration.precisions == profile.precisions)
            #expect(manifest.execution.capacity == 1024)
            #expect(manifest.execution.prefillLength == 145)
            #expect(manifest.execution.audioStartToken == 101)
            #expect(manifest.generation.inferenceTimesteps == 10)
            #expect(manifest.generation.samplesPerPatch == 7680)
            #expect(manifest.structure.sampleRate == 48_000)
            #expect(manifest.assets["dit"]?.functions["main"]?.inputs.map(\.name) == ["x", "mu", "t", "cond", "dt"])
        }
    }

    @Test(
        "VIN-03 The manifest check rejects a wrong format, model, configuration, precision, rate, patch or asset list",
        arguments: VoxCPMManifestDefect.allCases)
    func rejectsManifest(_ defect: VoxCPMManifestDefect) throws {
        let manifest = try VoxCPMBundleFixture.manifest(.fp32)
        let rewritten = try JSONSerialization.data(withJSONObject: JSONSerialization.jsonObject(with: manifest))
        #expect(try VoxCPMBundleInventory.profile(in: rewritten) == .fp32)
        let data = try Self.changed(manifest, defect)
        #expect(throws: SpeechError.self) { try VoxCPMBundleInventory.profile(in: data) }
        #expect(throws: SpeechError.self) { try VoxCPMBundleInventory.manifest(in: data) }
    }

    @Test("VIN-04 The bundled inventories hold 22 files with their totals and fp16_proj32 shares only proj with fp32")
    func bundledInventories() throws {
        let totals: [VoxCPMBundleProfile: Int64] = [
            .fp32: 9_349_325_520, .fp16: 4_677_413_664, .fp16Proj32: 4_715_188_166,
        ]
        for profile in VoxCPMBundleProfile.allCases {
            let files = try VoxCPMBundleInventory.files(profile: profile)
            #expect(files.count == 22, "\(profile)")
            #expect(files.reduce(0) { $0 + $1.bytes } == totals[profile], "\(profile)")
            #expect(files.first { $0.path == "tokenizer.json" }?.sha256 == VoxCPMTokenizer.tokenizerSHA256)
            #expect(files.contains { $0.path == VoxCPMBundleInventory.manifestPath }, "\(profile)")
        }
        let fp32 = try VoxCPMBundleInventory.files(profile: .fp32)
        let shared = try VoxCPMBundleInventory.files(profile: .fp16Proj32).filter {
            $0.path.hasPrefix("artifacts/") && fp32.contains($0)
        }
        #expect(
            shared.map(\.path)
                == ["main.hash", "main.mlirb", "metadata.json"].map { "artifacts/proj-fp32.aimodel/" + $0 })
    }

    @Test(
        "VIN-04 An inventory with a wrong format, count, total, digest, size or path is rejected",
        arguments: VoxCPMInventoryDefect.allCases)
    func rejectsInventory(_ defect: VoxCPMInventoryDefect) throws {
        let fixture = try VoxCPMBundleFixture()
        defer { fixture.remove() }
        let valid = try fixture.inventory()
        #expect(try VoxCPMBundleInventory.files(inventory: valid).count == fixture.contents.count)
        let data = try Self.changed(inventory: valid, defect)
        #expect(throws: SpeechError.self) { try VoxCPMBundleInventory.files(inventory: data) }
    }

    static func changed(_ data: Data, _ defect: VoxCPMManifestDefect) throws -> Data {
        if defect == .notJSON { return data.prefix(data.count / 2) }
        var json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        switch defect {
        case .format: json["format"] = 2
        case .modelID: try set(&json, ["model", "model_id"], "openbmb/VoxCPM")
        case .revision: try set(&json, ["model", "revision"], String(repeating: "0", count: 40))
        case .configurationName: try set(&json, ["configuration", "name"], "fp8")
        case .configurationPrecision: try set(&json, ["configuration", "precisions", "proj"], "fp16")
        case .assetPrecision: try set(&json, ["assets", "dit", "precision"], "fp16")
        case .sampleRate: try set(&json, ["structure", "sample_rate"], 44_100)
        case .patch: try set(&json, ["structure", "patch"], [1, 64, 8])
        case .tokenizer: json["tokenizer"] = "vocab.json"
        case .missingAsset: try set(&json, ["assets", "vae"], nil)
        case .extraAsset:
            let dit = try value(json, ["assets", "dit"])
            try set(&json, ["assets", "extra"], dit)
        case .assetPath: try set(&json, ["assets", "tslm", "path"], "artifacts/tslm-fp16.aimodel")
        case .missingFunction: try set(&json, ["assets", "tslm", "functions", "step"], nil)
        case .extraFunction:
            let step = try value(json, ["assets", "tslm", "functions", "step"])
            try set(&json, ["assets", "tslm", "functions", "decode"], step)
        case .notJSON: break
        }
        return try JSONSerialization.data(withJSONObject: json)
    }

    static func changed(inventory data: Data, _ defect: VoxCPMInventoryDefect) throws -> Data {
        if defect == .notJSON { return data.prefix(data.count / 2) }
        var json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        var files = try #require(json["files"] as? [[String: Any]])
        let count = try #require(json["count"] as? Int)
        let total = try #require(json["bytes"] as? Int)
        switch defect {
        case .format: json["format"] = 2
        case .count: json["count"] = count + 1
        case .total: json["bytes"] = total + 1
        case .digest: files[0]["sha256"] = "00"
        case .negativeSize:
            json["bytes"] = total - (files[0]["bytes"] as? Int ?? 0) - 1
            files[0]["bytes"] = -1
        case .duplicatePath:
            files[1]["path"] = files[0]["path"]
        case .absolutePath: files[0]["path"] = "/LICENSE"
        case .parentPath: files[0]["path"] = "artifacts/../LICENSE"
        case .combiningParentPath: files[0]["path"] = "../\u{301}LICENSE"
        case .nulPath: files[0]["path"] = "LICENSE\u{0}x"
        case .combiningName: files[0]["path"] = "artifacts/\u{301}x/main.mlirb"
        case .spacePath: files[0]["path"] = "LICENSE x"
        case .missingManifest:
            let index = try #require(files.firstIndex { $0["path"] as? String == "manifest.json" })
            json["bytes"] = total - (files[index]["bytes"] as? Int ?? 0)
            json["count"] = count - 1
            files.remove(at: index)
        case .notJSON: break
        }
        json["files"] = files
        return try JSONSerialization.data(withJSONObject: json)
    }

    static func value(_ object: [String: Any], _ path: [String]) throws -> Any {
        guard let key = path.first else { return object }
        let child = try #require(object[key])
        guard path.count > 1 else { return child }
        return try value(try #require(child as? [String: Any]), Array(path.dropFirst()))
    }

    static func set(_ object: inout [String: Any], _ path: [String], _ value: Any?) throws {
        guard let key = path.first else { return }
        guard path.count > 1 else {
            object[key] = value
            return
        }
        var child = try #require(object[key] as? [String: Any])
        try set(&child, Array(path.dropFirst()), value)
        object[key] = child
    }
}
