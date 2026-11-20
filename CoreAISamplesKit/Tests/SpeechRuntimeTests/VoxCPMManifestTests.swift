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
            #expect(manifest.format == (profile == .referenceFP32 ? 2 : 1), "\(profile)")
            #expect(Set(manifest.assets.keys) == Set(profile.assetNames), "\(profile)")
            #expect((manifest.reference != nil) == (profile == .referenceFP32), "\(profile)")
        }
    }

    @Test("VRA-01 Each configuration owns its asset order and the reference FP32 settings decode all seven values")
    func profileOrder() throws {
        let base = ["tslm", "ralm", "proj", "locenc", "dit", "vae"]
        #expect(VoxCPMBundleInventory.assetNames == base)
        for profile in [VoxCPMBundleProfile.fp32, .fp16, .fp16Proj32] {
            #expect(profile.assetNames == base, "\(profile)")
            #expect(try VoxCPMBundleInventory.manifest(in: VoxCPMBundleFixture.manifest(profile)).reference == nil)
        }
        let order = base + ["refenc"]
        let precisions = Dictionary(uniqueKeysWithValues: order.map { ($0, "fp32") })
        #expect(VoxCPMBundleProfile.referenceFP32.assetNames == order)
        #expect(VoxCPMBundleProfile.referenceFP32.precisions == precisions)
        let manifest = try VoxCPMBundleInventory.manifest(in: VoxCPMBundleFixture.manifest(.referenceFP32))
        let reference = try #require(manifest.reference)
        #expect(manifest.format == 2 && reference == VoxCPMReferenceConfiguration.pinned)
        #expect(reference.sampleRate == 16_000 && reference.minimumSamples == 80_000)
        #expect(reference.maximumSamples == 160_000 && reference.hopSamples == 640)
        #expect(reference.startToken == 103 && reference.endToken == 104 && reference.padding == "right_zero")
        #expect(VoxCPMReferenceConfiguration.paddedSamples == 161_280)
        #expect(VoxCPMReferenceConfiguration.paddedSamples % reference.hopSamples == 0)
    }

    @Test(
        "VRA-01 A reference manifest with a wrong reference value, format, configuration or encoder is rejected",
        arguments: [
            "changed:sample_rate", "changed:minimum_samples", "changed:maximum_samples", "changed:hop_samples",
            "changed:start_token", "changed:end_token", "changed:padding", "missing:sample_rate",
            "missing:minimum_samples", "missing:maximum_samples", "missing:hop_samples", "missing:start_token",
            "missing:end_token", "missing:padding", "null", "omitted", "format", "legacyName", "halfEncoder",
            "halfDeclaredEncoder", "encoderPath", "noEncoder", "extraEncoderFunction", "noEncoderFunction",
            "extraAsset", "halfModel",
        ])
    func rejectsReferenceManifest(_ defect: String) throws {
        let manifest = try VoxCPMBundleFixture.manifest(.referenceFP32)
        let rewritten = try JSONSerialization.data(withJSONObject: JSONSerialization.jsonObject(with: manifest))
        #expect(try VoxCPMBundleInventory.profile(in: rewritten) == .referenceFP32)
        let data = try Self.changed(reference: manifest, defect)
        #expect(throws: SpeechError.self) { try VoxCPMBundleInventory.profile(in: data) }
        #expect(throws: SpeechError.self) { try VoxCPMBundleInventory.manifest(in: data) }
    }

    @Test(
        "VRA-01 A legacy manifest that carries the reference settings or claims format 2 is rejected",
        arguments: [VoxCPMBundleProfile.fp32, .fp16, .fp16Proj32])
    func rejectsLegacyReference(_ profile: VoxCPMBundleProfile) throws {
        let reference = try #require(
            try JSONSerialization.jsonObject(with: VoxCPMBundleFixture.manifest(.referenceFP32)) as? [String: Any])
        for change in ["reference", "format", "both"] {
            var json = try #require(
                try JSONSerialization.jsonObject(with: VoxCPMBundleFixture.manifest(profile)) as? [String: Any])
            if change != "format" { json["reference"] = reference["reference"] }
            if change != "reference" { json["format"] = 2 }
            let data = try JSONSerialization.data(withJSONObject: json)
            #expect(throws: SpeechError.self, "\(change)") { try VoxCPMBundleInventory.manifest(in: data) }
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
        for profile in [VoxCPMBundleProfile.fp32, .fp16, .fp16Proj32] {
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

    @Test("VRA-01 The reference FP32 inventory lists 25 files that keep the FP32 payload and add the encoder")
    func referenceInventory() throws {
        let reference = try VoxCPMBundleInventory.files(profile: .referenceFP32)
        let fp32 = try VoxCPMBundleInventory.files(profile: .fp32)
        #expect(reference.count == 25 && reference.reduce(0) { $0 + $1.bytes } == 9_541_520_509)
        #expect(
            reference.filter { !fp32.contains($0) }.map(\.path) == [
                "NOTICE", "artifacts/refenc-fp32.aimodel/main.hash", "artifacts/refenc-fp32.aimodel/main.mlirb",
                "artifacts/refenc-fp32.aimodel/metadata.json", "manifest.json",
            ])
        #expect(fp32.filter { !reference.contains($0) }.map(\.path) == ["NOTICE", "manifest.json"])
        let manifest = try #require(reference.first { $0.path == VoxCPMBundleInventory.manifestPath })
        #expect(manifest.sha256 == "5beadfe31f8b37667f3e4ef42a8451b29dca1814a4cb53a253e96d9a2fdd1b51")
        let data = try VoxCPMBundleFixture.manifest(.referenceFP32)
        #expect(Int64(data.count) == manifest.bytes && VoxCPMBundleFixture.digest(data) == manifest.sha256)
        #expect(reference.first { $0.path == "tokenizer.json" }?.sha256 == VoxCPMTokenizer.tokenizerSHA256)
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

    static func changed(reference data: Data, _ defect: String) throws -> Data {
        var json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let parts = defect.split(separator: ":").map(String.init)
        switch parts[0] {
        case "changed":
            let path = ["reference", parts[1]]
            let old = try value(json, path)
            try set(&json, path, (old as? Int).map { $0 + 1 } ?? "reflect")
        case "missing": try set(&json, ["reference", parts[1]], nil)
        case "null": json["reference"] = NSNull()
        case "omitted": json["reference"] = nil
        case "format": json["format"] = 1
        case "legacyName": try set(&json, ["configuration", "name"], "fp32")
        case "halfEncoder": try set(&json, ["configuration", "precisions", "refenc"], "fp16")
        case "halfDeclaredEncoder": try set(&json, ["assets", "refenc", "precision"], "fp16")
        case "encoderPath": try set(&json, ["assets", "refenc", "path"], "artifacts/locenc-fp32.aimodel")
        case "noEncoder": try set(&json, ["assets", "refenc"], nil)
        case "extraEncoderFunction":
            let main = try value(json, ["assets", "refenc", "functions", "main"])
            try set(&json, ["assets", "refenc", "functions", "aux"], main)
        case "noEncoderFunction": try set(&json, ["assets", "refenc", "functions", "main"], nil)
        case "extraAsset":
            let encoder = try value(json, ["assets", "refenc"])
            try set(&json, ["assets", "extra"], encoder)
        default:
            try set(&json, ["configuration", "precisions", "dit"], "fp16")
            try set(&json, ["assets", "dit", "precision"], "fp16")
            try set(&json, ["assets", "dit", "path"], "artifacts/dit-fp16.aimodel")
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
