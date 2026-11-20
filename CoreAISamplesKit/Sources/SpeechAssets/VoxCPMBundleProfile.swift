public enum VoxCPMBundleProfile: String, CaseIterable, Sendable {
    case fp32
    case fp16
    case fp16Proj32 = "fp16_proj32"
    case referenceFP32 = "reference_fp32"

    public var assetNames: [String] {
        switch self {
        case .fp32, .fp16, .fp16Proj32: VoxCPMBundleInventory.assetNames
        case .referenceFP32: VoxCPMBundleInventory.assetNames + ["refenc"]
        }
    }

    public var precisions: [String: String] {
        let precision = self == .fp32 || self == .referenceFP32 ? "fp32" : "fp16"
        var result = Dictionary(uniqueKeysWithValues: assetNames.map { ($0, precision) })
        if self == .fp16Proj32 { result["proj"] = "fp32" }
        return result
    }

    var format: Int {
        switch self {
        case .fp32, .fp16, .fp16Proj32: 1
        case .referenceFP32: 2
        }
    }

    var reference: VoxCPMReferenceConfiguration? {
        switch self {
        case .fp32, .fp16, .fp16Proj32: nil
        case .referenceFP32: .pinned
        }
    }
}
