import Foundation
import SpeechCore
import Testing

@testable import SpeechAssets
@testable import SpeechRuntime

struct VoxCPMContractTests {
    typealias Tensor = VoxCPMSignature.Tensor

    static func observed(_ manifest: VoxCPMBundleManifest, _ asset: String) -> ([String], [String: VoxCPMSignature]) {
        let functions = manifest.assets[asset]?.functions ?? [:]
        return (functions.keys.sorted(), functions.mapValues(VoxCPMSignature.init))
    }

    static func manifest(_ profile: VoxCPMBundleProfile = .fp32, _ change: (inout [String: Any]) throws -> Void)
        throws -> VoxCPMBundleManifest
    {
        var json = try #require(
            try JSONSerialization.jsonObject(with: VoxCPMBundleFixture.manifest(profile)) as? [String: Any])
        try change(&json)
        return try VoxCPMBundleInventory.manifest(in: JSONSerialization.data(withJSONObject: json))
    }

    static func reshape(
        _ json: inout [String: Any], _ asset: String, _ functions: [String], _ kind: String, _ name: String,
        _ shape: [Int]
    ) throws {
        for function in functions {
            let path = ["assets", asset, "functions", function, kind]
            var entries = try #require(try VoxCPMManifestTests.value(json, path) as? [[Any]])
            let index = try #require(entries.firstIndex { $0.first as? String == name })
            entries[index][2] = shape
            try VoxCPMManifestTests.set(&json, path, entries)
        }
    }

    static func changed(_ signature: VoxCPMSignature?, _ change: (inout VoxCPMSignature) -> Void) throws
        -> VoxCPMSignature
    {
        var result = try #require(signature)
        change(&result)
        return result
    }

    @Test("VPL-04 Every bundled configuration passes when the asset descriptors equal the manifest contract")
    func bundledContracts() throws {
        for profile in VoxCPMBundleProfile.allCases {
            let manifest = try VoxCPMBundleInventory.manifest(in: VoxCPMBundleFixture.manifest(profile))
            let contract = try VoxCPMContract(manifest: manifest)
            #expect(contract.profile == profile)
            #expect(contract.prefillLength == 145 && contract.capacity == 1024 && contract.samplesPerPatch == 7680)
            for asset in VoxCPMBundleInventory.assetNames {
                let (names, signatures) = Self.observed(manifest, asset)
                try contract.check(asset: asset, functionNames: names, observed: signatures)
            }
        }
    }

    @Test(
        "VPL-04 An asset whose function or descriptor differs from the contract is rejected before any call",
        arguments: VoxCPMDescriptorDefect.allCases)
    func rejectsDescriptor(_ defect: VoxCPMDescriptorDefect) throws {
        let manifest = try VoxCPMBundleInventory.manifest(in: VoxCPMBundleFixture.manifest(.fp32))
        let contract = try VoxCPMContract(manifest: manifest)
        let asset: String
        switch defect {
        case .inputType, .missingFunction: asset = "tslm"
        case .outputShape, .reorderedInputs: asset = "dit"
        case .missingState: asset = "vae"
        case .stateShape, .extraFunction: asset = "ralm"
        case .extraInput: asset = "proj"
        case .renamedOutput: asset = "locenc"
        }
        var (names, signatures) = Self.observed(manifest, asset)
        switch defect {
        case .inputType:
            signatures["step"] = try Self.changed(signatures["step"]) {
                $0 = VoxCPMSignature(
                    inputs: $0.inputs.map {
                        $0.name == "position" ? Tensor(name: $0.name, type: "float32", shape: $0.shape) : $0
                    },
                    outputs: $0.outputs, states: $0.states)
            }
        case .outputShape:
            signatures["main"] = try Self.changed(signatures["main"]) {
                $0 = VoxCPMSignature(
                    inputs: $0.inputs, outputs: [Tensor(name: "velocity", type: "float32", shape: [2, 64, 8])],
                    states: $0.states)
            }
        case .missingState:
            signatures["main"] = try Self.changed(signatures["main"]) {
                $0 = VoxCPMSignature(inputs: $0.inputs, outputs: $0.outputs, states: Array($0.states.dropLast()))
            }
        case .stateShape:
            signatures["step"] = try Self.changed(signatures["step"]) {
                $0 = VoxCPMSignature(
                    inputs: $0.inputs, outputs: $0.outputs,
                    states: $0.states.map { Tensor(name: $0.name, type: $0.type, shape: [8, 2, 512, 128]) })
            }
        case .extraInput:
            signatures["heads"] = try Self.changed(signatures["heads"]) {
                $0 = VoxCPMSignature(
                    inputs: $0.inputs + [Tensor(name: "extra", type: "float32", shape: [1])], outputs: $0.outputs,
                    states: $0.states)
            }
        case .renamedOutput:
            signatures["main"] = try Self.changed(signatures["main"]) {
                $0 = VoxCPMSignature(
                    inputs: $0.inputs, outputs: [Tensor(name: "embed", type: "float32", shape: [1, 2048])],
                    states: $0.states)
            }
        case .reorderedInputs:
            signatures["main"] = try Self.changed(signatures["main"]) {
                $0 = VoxCPMSignature(inputs: $0.inputs.reversed(), outputs: $0.outputs, states: $0.states)
            }
        case .extraFunction:
            names.append("decode")
            signatures["decode"] = signatures["step"]
        case .missingFunction:
            names.removeAll { $0 == "embed" }
            signatures["embed"] = nil
        }
        #expect(throws: SpeechError.self) {
            try contract.check(asset: asset, functionNames: names, observed: signatures)
        }
    }

    @Test(
        "VPL-04 A manifest function contract that the pipeline does not expect is rejected even when the asset matches it"
    )
    func rejectsUnexpectedContract() throws {
        let cases: [(String, (inout [String: Any]) throws -> Void)] = [
            ("dit", { try Self.reshape(&$0, "dit", ["main"], "inputs", "t", [3]) }),
            ("tslm", { try Self.reshape(&$0, "tslm", ["prefill", "step"], "states", "k_cache", [28, 2, 512, 128]) }),
            ("vae", { try Self.reshape(&$0, "vae", ["main"], "outputs", "audio", [1, 1, 4000]) }),
            ("proj", { try Self.reshape(&$0, "proj", ["heads"], "outputs", "stop_logits", [1, 3]) }),
        ]
        for (asset, change) in cases {
            let manifest = try Self.manifest(.fp32, change)
            let contract = try VoxCPMContract(manifest: manifest)
            let (names, signatures) = Self.observed(manifest, asset)
            #expect(throws: SpeechError.self, "\(asset)") {
                try contract.check(asset: asset, functionNames: names, observed: signatures)
            }
        }
    }

    @Test("VPL-04 The prefill and step functions must report the same full descriptor for every shared state")
    func sharedStateLayouts() throws {
        let layout = ["k_cache": [262_144, 131_072, 128, 1], "v_cache": [262_144, 131_072, 128, 1]]
        try VoxCPMContract.checkSharedStates(asset: "ralm", ["prefill": layout, "step": layout])
        let stateless: [String: [String: [Int]]] = ["heads": [:], "fuse": [:]]
        try VoxCPMContract.checkSharedStates(asset: "proj", stateless)
        let cases: [[String: [String: [Int]]]] = [
            ["prefill": layout, "step": layout.merging(["k_cache": [262_144, 131_072, 256, 1]]) { $1 }],
            ["prefill": layout, "step": layout.filter { $0.key == "k_cache" }],
            ["prefill": layout, "step": layout.merging(["w_cache": [1]]) { $1 }],
            ["prefill": [:], "step": [:]],
        ]
        for states in cases {
            #expect(throws: SpeechError.self) { try VoxCPMContract.checkSharedStates(asset: "ralm", states) }
        }
    }

    @Test(
        "VPL-04 A manifest whose generation or execution values differ from the VoxCPMCore assumptions is rejected",
        arguments: VoxCPMGenerationDefect.allCases)
    func rejectsGeneration(_ defect: VoxCPMGenerationDefect) throws {
        let manifest = try Self.manifest { try VoxCPMManifestTests.set(&$0, defect.change.path, defect.change.value) }
        #expect(throws: SpeechError.self) { try VoxCPMContract(manifest: manifest) }
    }
}
