#if os(iOS)
    import Darwin
    import Foundation
    import ImageGenerationCore
    import ModelLibraryFeature

    enum ImageGenerationDeviceProfiles {
        private struct Calibration {
            let hardwareModel: String
            let osBuild: String
            let budget: ImageMemoryBudget
        }

        private static let calibrations: [Calibration] = []

        static func budget(for resource: ValidatedModelResource) -> ImageMemoryBudget? {
            guard let hardware = systemValue("hw.machine"), let build = systemValue("kern.osversion") else {
                return nil
            }
            return calibrations.first {
                $0.hardwareModel == hardware && $0.osBuild == build && $0.budget.manifestSHA256 == resource.revision
            }?.budget
        }

        private static func systemValue(_ name: String) -> String? {
            var size = 0
            guard sysctlbyname(name, nil, &size, nil, 0) == 0, (1...4_096).contains(size) else { return nil }
            var bytes = [UInt8](repeating: 0, count: size)
            let result = bytes.withUnsafeMutableBytes { sysctlbyname(name, $0.baseAddress, &size, nil, 0) }
            guard result == 0 else { return nil }
            return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
        }
    }
#endif
