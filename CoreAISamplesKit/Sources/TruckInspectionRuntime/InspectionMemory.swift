import Darwin
import Foundation

enum InspectionMemory {
    static func snapshot() -> String {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return "memory: unavailable code=\(result)" }
        let peak = info.ledger_phys_footprint_peak > 0 ? String(info.ledger_phys_footprint_peak) : "unavailable"
        var text =
            "rssBytes=\(info.resident_size) peakRSSBytes=\(info.resident_size_peak) "
            + "footprintBytes=\(info.phys_footprint) processPeakFootprintBytes=\(peak)"
        #if os(iOS)
            text += " availableBytes=\(os_proc_available_memory())"
        #endif
        return text
    }

    static func environment() -> String {
        var size = 0
        let name = "hw.machine"
        var hardware = "unknown"
        if sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 {
            var value = [CChar](repeating: 0, count: size)
            if sysctlbyname(name, &value, &size, nil, 0) == 0 {
                hardware = String(decoding: value.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
            }
        }
        let process = ProcessInfo.processInfo
        return "device=\(hardware) os=\(process.operatingSystemVersionString) "
            + "thermal=\(process.thermalState.rawValue) lowPower=\(process.isLowPowerModeEnabled)"
    }
}
