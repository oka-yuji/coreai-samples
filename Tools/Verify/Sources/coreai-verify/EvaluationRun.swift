import CryptoKit
import Foundation

struct EvaluationRun: Codable, Equatable {
    struct Contract: Codable, Equatable {
        let version: String
        let sha256: String
    }

    struct Asset: Codable, Equatable {
        let name: String
        let mainHash: String
        let sourceModelId: String?
    }

    struct Implementation: Codable, Equatable {
        var command = "coreai-verify evaluate"
        let commit: String
        var osVersion: String = {
            let version = ProcessInfo.processInfo.operatingSystemVersion
            return "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
        }()
        var osBuild = TypedDecideHost.sysctl("kern.osversion")
    }

    struct CallShape: Codable, Equatable {
        let prefillChunk: Int
        let prefillFunction: Bool
        let expectFrequentReshapes: Bool?
        let requestedCompute: String?
        let contextLimit: Int
    }

    struct Inputs: Codable, Equatable {
        let sha256: String
        let count: Int
    }

    struct Load: Codable, Equatable {
        let cached: Bool
        let specializeS: Double
        let loadS: Double
    }

    private struct Identity: Encodable {
        let id: String
        let contract: Contract
        let set: String
        let role: String
        let mainHash: String
        let inputs: Inputs
        let callShape: CallShape
        let implementation: Implementation
    }

    let id: String
    let contract: Contract
    let set: String
    let role: String
    let asset: Asset
    let implementation: Implementation
    let callShape: CallShape
    let inputs: Inputs
    var items: Int
    let started: Date
    var finished: Date?
    let load: Load

    func differences(from stored: EvaluationRun) -> [String] {
        [
            ("contract", contract == stored.contract), ("set", set == stored.set), ("role", role == stored.role),
            ("asset.main_hash", asset.mainHash == stored.asset.mainHash), ("inputs", inputs == stored.inputs),
            ("call_shape", callShape == stored.callShape), ("implementation", implementation == stored.implementation),
        ].filter { !$0.1 }.map(\.0)
    }

    func identityDigest() throws -> String {
        let identity = Identity(
            id: id, contract: contract, set: set, role: role, mainHash: asset.mainHash, inputs: inputs,
            callShape: callShape,
            implementation: implementation)
        return hex(Data(SHA256.hash(data: try JSONEncoder.records.encode(identity))))
    }
}
