public enum VoxCPMBundleProfile: String, CaseIterable, Sendable {
    case fp32
    case fp16
    case fp16Proj32 = "fp16_proj32"

    public var precisions: [String: String] {
        let precision = self == .fp32 ? "fp32" : "fp16"
        var result = Dictionary(uniqueKeysWithValues: VoxCPMBundleInventory.assetNames.map { ($0, precision) })
        if self == .fp16Proj32 { result["proj"] = "fp32" }
        return result
    }
}
