#if os(iOS)
    import CoreAI
    import Darwin
    import Foundation
    import UIKit
    import os

    @MainActor
    final class LivePhoneCheckEnvironment: PhoneCheckEnvironment {
        private var previousTimer = false
        private var previousBattery = false
        func begin() {
            previousTimer = UIApplication.shared.isIdleTimerDisabled
            previousBattery = UIDevice.current.isBatteryMonitoringEnabled
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        }
        func end() {
            UIApplication.shared.isIdleTimerDisabled = previousTimer
            UIDevice.current.isBatteryMonitoringEnabled = previousBattery
        }
        func info() -> PhoneDeviceInfo {
            var value = utsname()
            let result = uname(&value)
            let hardware = withUnsafeBytes(of: &value.machine) {
                String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self)
            }
            return PhoneDeviceInfo(
                osVersion: UIDevice.current.systemVersion,
                osBuild: ProcessInfo.processInfo.operatingSystemVersionString,
                hardware: result == 0 ? hardware : "unknown", batteryState: UIDevice.current.batteryState.rawValue,
                lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                architecture: AIModel.deviceArchitectureName)
        }
        func memory() -> PhoneMemory {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let result = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                }
            }
            return PhoneMemory(
                availableBytes: UInt64(os_proc_available_memory()),
                footprintBytes: result == KERN_SUCCESS ? info.phys_footprint : nil,
                peakFootprintBytes: result == KERN_SUCCESS ? info.ledger_phys_footprint_peak : nil,
                thermalState: ProcessInfo.processInfo.thermalState.rawValue)
        }
    }
#endif
