import Darwin
import Foundation
import Metal

struct ResidencyReport: Encodable {
    let scope = "Real EmbeddingModel through ModelLibraryRunner; macOS test process, not iPhone UI acceptance"
    let os = ProcessInfo.processInfo.operatingSystemVersionString
    let artifactID: String
    let revision: String
    let loads: Int
    let nativeUnloads: Int
    let samples: [ResidencyMemory]
}

struct ResidencyMemory: Encodable {
    let phase: String
    let status: Int32
    let rssBytes: UInt64?
    let footprintBytes: UInt64?
    let graphicsFootprintBytes: UInt64?
    let kernelLifetimePeakBytes: UInt64?
    let visibleMetalAllocatedBytes: Int?
    let metalScope = "currentAllocatedSize on the process default MTLDevice; not total GPU memory"
    let boundary = "point observation; neither a phase peak nor a sum of independent RAM allocations"

    static func capture(_ phase: String) -> Self {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return Self(
            phase: phase, status: status,
            rssBytes: status == KERN_SUCCESS ? info.resident_size : nil,
            footprintBytes: status == KERN_SUCCESS ? info.phys_footprint : nil,
            graphicsFootprintBytes: status == KERN_SUCCESS
                ? UInt64(max(0, info.ledger_tag_graphics_footprint))
                    + UInt64(max(0, info.ledger_tag_graphics_footprint_compressed)) : nil,
            kernelLifetimePeakBytes: status == KERN_SUCCESS ? UInt64(max(0, info.ledger_phys_footprint_peak)) : nil,
            visibleMetalAllocatedBytes: MTLCreateSystemDefaultDevice()?.currentAllocatedSize)
    }
}
