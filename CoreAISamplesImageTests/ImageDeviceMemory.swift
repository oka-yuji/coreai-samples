import Darwin
import Foundation
import ImageGenerationCore
import Metal

enum ImageDeviceMemory {
    static func capture() -> ImageMemoryObservation {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return ImageMemoryObservation(
            scope: .processDirtyBudget, availableBytes: UInt64(os_proc_available_memory()),
            footprintBytes: status == KERN_SUCCESS ? info.phys_footprint : nil,
            residentBytes: status == KERN_SUCCESS ? info.resident_size : nil,
            kernelLifetimePeakBytes: status == KERN_SUCCESS ? UInt64(max(0, info.ledger_phys_footprint_peak)) : nil,
            visibleMetalBytes: MTLCreateSystemDefaultDevice().map { UInt64($0.currentAllocatedSize) },
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            thermalState: ProcessInfo.processInfo.thermalState.rawValue,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
    }

    static func systemValue(_ name: String) throws -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, (1...4_096).contains(size) else {
            throw ImageExecutionError.invalidResource
        }
        var bytes = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { throw ImageExecutionError.invalidResource }
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }
}

@MainActor
final class ImageDeviceMemoryObserver {
    struct Window: Encodable {
        let label: String
        let startedSeconds: Double
        let before: ImageMemoryObservation
        var endedSeconds: Double
        var after: ImageMemoryObservation
        var sampleCount = 0
        var invalidSampleCount = 0
        var maximumGapSeconds = 0.0
        var sampledFootprintMaximum: UInt64?
        var sampledResidentMaximum: UInt64?
        var sampledMetalMaximum: UInt64?
        var minimumAvailableBytes: UInt64?
        var kernelLifetimePeakMaximum: UInt64?

        mutating func append(_ point: ImageMemoryObservation, at seconds: Double) {
            maximumGapSeconds = max(maximumGapSeconds, seconds - endedSeconds)
            endedSeconds = seconds
            after = point
            sampleCount += 1
            if point.footprintBytes == nil || point.residentBytes == nil || point.visibleMetalBytes == nil
                || point.kernelLifetimePeakBytes == nil || point.availableBytes == nil
            {
                invalidSampleCount += 1
            }
            sampledFootprintMaximum = maximum(sampledFootprintMaximum, point.footprintBytes)
            sampledResidentMaximum = maximum(sampledResidentMaximum, point.residentBytes)
            sampledMetalMaximum = maximum(sampledMetalMaximum, point.visibleMetalBytes)
            kernelLifetimePeakMaximum = maximum(kernelLifetimePeakMaximum, point.kernelLifetimePeakBytes)
            minimumAvailableBytes = [minimumAvailableBytes, point.availableBytes].compactMap { $0 }.min()
        }

        private func maximum(_ first: UInt64?, _ second: UInt64?) -> UInt64? {
            [first, second].compactMap { $0 }.max()
        }
    }

    private let start = ContinuousClock.now
    private var active = true
    private var windows: [Window] = []
    private var current: Window?

    func transition(_ label: String) {
        let point = ImageDeviceMemory.capture()
        let seconds = (ContinuousClock.now - start) / .seconds(1)
        if var previous = current {
            previous.append(point, at: seconds)
            windows.append(previous)
        }
        var next = Window(
            label: label, startedSeconds: seconds, before: point, endedSeconds: seconds, after: point)
        next.append(point, at: seconds)
        current = next
    }

    func watch() async {
        await withTaskCancellationShield {
            while active {
                current?.append(ImageDeviceMemory.capture(), at: (ContinuousClock.now - start) / .seconds(1))
                do { try await Task.sleep(for: .milliseconds(10)) } catch { return }
            }
        }
    }

    func finish() -> [Window] {
        active = false
        if var last = current {
            last.append(ImageDeviceMemory.capture(), at: (ContinuousClock.now - start) / .seconds(1))
            windows.append(last)
            current = nil
        }
        return windows
    }
}
