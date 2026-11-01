#if os(iOS)
    import Darwin
    import Foundation
    import ImageGenerationCore
    import ImageGenerationRuntime
    import ModelLibraryFeature

    enum ImageGenerationDeviceProfiles {
        private static let calibrations: [ImageDeviceCalibration] =
            (try? ImageDeviceCalibration.makeQwenImage21NativeComplex()).map { [$0] } ?? []

        static func budget(for resource: ValidatedModelResource) -> ImageMemoryBudget? {
            guard let hardware = systemValue("hw.machine"), let build = systemValue("kern.osversion") else {
                return nil
            }
            return calibrations.lazy.compactMap {
                $0.budget(for: resource, hardwareModel: hardware, osBuild: build)
            }.first
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
