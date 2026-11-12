import Darwin
import Foundation

struct ForecastMemory: Encodable {
    let phase: String
    let status: Int32
    let returnedWords: UInt32
    let rssBytes: UInt64?
    let footprintBytes: UInt64?
    let kernelLifetimePeakBytes: UInt64?
    let boundary =
        "Current RSS and physical footprint are point samples. Kernel peak covers the whole test process lifetime, not a request delta or GPU allocation sum. Unsupported fields are null."

    static func capture(_ phase: String) -> Self {
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
            phase: phase, status: status, returnedWords: count,
            rssBytes: present(\.resident_size) ? info.resident_size : nil,
            footprintBytes: present(\.phys_footprint) ? info.phys_footprint : nil,
            kernelLifetimePeakBytes: present(\.ledger_phys_footprint_peak)
                ? UInt64(max(0, info.ledger_phys_footprint_peak)) : nil)
    }

    static func machine() -> String? {
        var size = 0
        guard sysctlbyname("hw.model", nil, &size, nil, 0) == 0, size > 0, size < 1024 else { return nil }
        var bytes = [UInt8](repeating: 0, count: size)
        guard sysctlbyname("hw.model", &bytes, &size, nil, 0) == 0 else { return nil }
        return String(decoding: bytes.prefix(while: { $0 != 0 }), as: UTF8.self)
    }

    enum CodingKeys: String, CodingKey {
        case phase, status, returnedWords, rssBytes, footprintBytes, kernelLifetimePeakBytes, boundary
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(phase, forKey: .phase)
        try container.encode(status, forKey: .status)
        try container.encode(returnedWords, forKey: .returnedWords)
        try container.encode(rssBytes, forKey: .rssBytes)
        try container.encode(footprintBytes, forKey: .footprintBytes)
        try container.encode(kernelLifetimePeakBytes, forKey: .kernelLifetimePeakBytes)
        try container.encode(boundary, forKey: .boundary)
    }
}
