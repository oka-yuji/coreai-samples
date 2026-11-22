import Darwin

enum ProcessFootprint {
    enum Failure: Error {
        case unavailable(kern_return_t)
        case incomplete(mach_msg_type_number_t)
    }

    static func bytes() throws -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { throw Failure.unavailable(result) }
        return info.phys_footprint
    }

    static func graphicsBytes() throws -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { throw Failure.unavailable(result) }
        guard let offset = MemoryLayout<task_vm_info_data_t>.offset(of: \.ledger_tag_graphics_footprint_compressed),
            Int(count) * MemoryLayout<integer_t>.size >= offset + MemoryLayout<Int64>.size
        else { throw Failure.incomplete(count) }
        return UInt64(max(0, info.ledger_tag_graphics_footprint))
            + UInt64(max(0, info.ledger_tag_graphics_footprint_compressed))
    }
}
