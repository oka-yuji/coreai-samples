import Foundation
import ImageGenerationCore
import Testing

struct ImageGenerationProfileTests {
    @Test("The pinned five-component profile preserves its real physical tensor boundaries")
    func originalProfileDecodes() throws {
        let profile = try ImageGenerationProfile.decode(Data(contentsOf: NativeGolden.resource("profile-v2.json")))
        #expect(profile.source.revision == ImageGenerationProfile.revision)
        #expect(Set(profile.components.keys) == ImageGenerationProfile.componentNames)
        #expect(profile.components["denoiser"]?.outputs.first?.shape == [1, 2_304, 64])
        #expect(profile.components["vae_decoder"]?.outputs.first?.shape == [1, 4, 1, 512, 512])
        #expect(profile.workload.schedulerDeltaDtype == "model-output")
    }

    @Test("Unsupported shapes dtypes source identities and scheduler semantics cannot pass profile validation")
    func profileRejectsContractChanges() throws {
        let original = try Data(contentsOf: NativeGolden.resource("profile-v2.json"))
        for (path, value) in [
            (["format"], 1), (["source", "revision"], String(repeating: "0", count: 40)),
            (["workload", "joint_capacity"], 1_536), (["workload", "batch"], true),
            (["workload", "scheduler_product_dtype"], "float32"),
            (["workload", "scheduler_compute_dtype"], "float16"),
            (["workload", "prefix_kv_cache"], true), (["components", "text", "asset"], "../escape.aimodel"),
            (["components", "vision", "compute_dtype"], "float64"),
            (["components", "text", "state_names"], ["kv"]),
        ] as [([String], Any)] {
            let root = try #require(JSONSerialization.jsonObject(with: original) as? [String: Any])
            let changed = replacing(root, at: path, value: value)
            let data = try JSONSerialization.data(withJSONObject: changed)
            #expect(throws: (any Error).self) { try ImageGenerationProfile.decode(data) }
        }
    }

    @Test("Bundle validation excludes its own hash and requires every component without trusting a passed flag")
    func bundleInventoryIsExplicit() throws {
        let data = try Data(contentsOf: NativeGolden.resource("profile-v2.json"))
        let profile = try ImageGenerationProfile.decode(data)
        let value = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var assets: [String: Any] = [:]
        for component in profile.components.values {
            assets[component.asset + "/main.mlirb"] = ["bytes": 100, "sha256": String(repeating: "a", count: 64)]
        }
        var manifest: [String: Any] = ["format": 2, "profile": value, "assets": assets, "passed": true]
        _ = try ImageGenerationManifest.decode(JSONSerialization.data(withJSONObject: manifest))
        assets["bundle.json"] = ["bytes": 10, "sha256": String(repeating: "b", count: 64)]
        manifest["assets"] = assets
        #expect(throws: ImageGenerationError.invalidProfile("bundle inventory")) {
            try ImageGenerationManifest.decode(JSONSerialization.data(withJSONObject: manifest))
        }
        assets["bundle.json"] = nil
        let vision = try #require(profile.components["vision"])
        assets[vision.asset + "/main.mlirb"] = nil
        manifest["assets"] = assets
        #expect(throws: ImageGenerationError.invalidProfile("missing component asset")) {
            try ImageGenerationManifest.decode(JSONSerialization.data(withJSONObject: manifest))
        }
    }

    @Test("Latent scaling preserves channel and token order and rejects invalid scale values")
    func scalingRoundTripsAndRejectsInvalidValues() throws {
        let scaling = try NativeGolden.load().scaling
        let values = (0..<ImageEuler.latentCount).map { Float($0 % 64) / 16 - 2 }
        let channels = try scaling.denormalize(values)
        #expect(channels[1_024] == values[1] * scaling.standardDeviation[1] + scaling.mean[1])
        let restored = try scaling.normalize(channels)
        let maximum = zip(values, restored).map { abs($0 - $1) }.max() ?? .infinity
        #expect(maximum < 0.000_001)
        #expect(throws: ImageGenerationError.invalidProfile("latent scaling")) {
            try ImageLatentScaling(mean: [0], standardDeviation: [0])
        }
        #expect(throws: ImageGenerationError.invalidShape) { try scaling.normalize([0]) }
    }

    private func replacing(_ object: [String: Any], at path: [String], value: Any) -> [String: Any] {
        guard let key = path.first else { return object }
        var result = object
        if path.count == 1 {
            result[key] = value
        } else if let child = result[key] as? [String: Any] {
            result[key] = replacing(child, at: Array(path.dropFirst()), value: value)
        }
        return result
    }
}
