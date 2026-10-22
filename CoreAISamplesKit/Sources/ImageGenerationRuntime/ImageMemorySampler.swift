import Darwin
import Foundation
import ImageGenerationCore
import Metal

enum ImageMemorySampler {
    static func capture() -> ImageMemoryObservation {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        let available: UInt64?
        let scope: ImageMemoryScope
        #if os(iOS)
            available = UInt64(os_proc_available_memory())
            scope = .processDirtyBudget
        #else
            available = systemHeadroom()
            scope = available == nil ? .unavailable : .estimatedSystemHeadroom
        #endif
        let metal = MTLCreateSystemDefaultDevice()?.currentAllocatedSize
        return ImageMemoryObservation(
            scope: scope, availableBytes: available,
            footprintBytes: status == KERN_SUCCESS ? info.phys_footprint : nil,
            residentBytes: status == KERN_SUCCESS ? info.resident_size : nil,
            kernelLifetimePeakBytes: status == KERN_SUCCESS ? UInt64(max(0, info.ledger_phys_footprint_peak)) : nil,
            visibleMetalBytes: metal.flatMap { $0 >= 0 ? UInt64($0) : nil },
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            thermalState: ProcessInfo.processInfo.thermalState.rawValue,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
    }

    #if os(macOS)
        private static func systemHeadroom() -> UInt64? {
            let host = mach_host_self()
            defer { mach_port_deallocate(mach_task_self_, host) }
            var info = vm_statistics64_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
            let status = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    host_statistics64(host, HOST_VM_INFO64, $0, &count)
                }
            }
            var pageSize: vm_size_t = 0
            guard status == KERN_SUCCESS, host_page_size(host, &pageSize) == KERN_SUCCESS, pageSize > 0 else {
                return nil
            }
            let pages = UInt64(info.free_count) + UInt64(info.inactive_count)
            let (bytes, overflow) = pages.multipliedReportingOverflow(by: UInt64(pageSize))
            return overflow ? nil : bytes
        }
    #endif
}
