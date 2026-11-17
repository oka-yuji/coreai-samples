import Darwin
import Foundation

struct VoxCPMMemory: Sendable {
    let footprintBytes: UInt64?
    let residentBytes: UInt64?
    let lifetimePeakBytes: UInt64?

    static func capture() -> Self {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        func present<T>(_ field: KeyPath<task_vm_info_data_t, T>) -> Bool {
            guard status == KERN_SUCCESS, let offset = MemoryLayout<task_vm_info_data_t>.offset(of: field) else {
                return false
            }
            return Int(count) * MemoryLayout<integer_t>.size >= offset + MemoryLayout<T>.size
        }
        return Self(
            footprintBytes: present(\.phys_footprint) ? info.phys_footprint : nil,
            residentBytes: present(\.resident_size) ? info.resident_size : nil,
            lifetimePeakBytes: present(\.ledger_phys_footprint_peak)
                ? UInt64(max(0, info.ledger_phys_footprint_peak)) : nil)
    }

    var json: [String: Any] {
        [
            "footprintBytes": footprintBytes.map { $0 as Any } ?? NSNull(),
            "residentBytes": residentBytes.map { $0 as Any } ?? NSNull(),
            "lifetimePeakBytes": lifetimePeakBytes.map { $0 as Any } ?? NSNull(),
        ]
    }
}
