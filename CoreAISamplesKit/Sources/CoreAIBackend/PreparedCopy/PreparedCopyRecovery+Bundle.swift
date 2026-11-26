import CoreAI
import Darwin
import Foundation

extension PreparedCopyRecovery {
    static func forBundle(_ bundle: ModelBundle) throws -> PreparedCopyRecovery? {
        guard let hash = bundle.mainHash, hash.utf8.count == 64 else { return nil }
        guard let process = PreparedCopyProcess.current() else { throw POSIXError(.ESRCH) }
        let options = Engine.specializationOptions()
        let compute = Engine.computeUnitName(options.preferredComputeUnitKind)
        let key = PreparedCopyKey(
            namespace: Bundle.main.bundleIdentifier ?? ProcessInfo.processInfo.processName,
            osBuild: try osBuild(), assetHash: hash,
            options: "compute=\(compute);frequentReshapes=\(options.expectFrequentReshapes)")
        let directory = URL.applicationSupportDirectory.appending(path: "coreai-samples/PreparedCopy")
        return PreparedCopyRecovery(directory: directory, key: key, process: process)
    }

    private static func osBuild() throws -> String {
        var count = 0
        guard sysctlbyname("kern.osversion", nil, &count, nil, 0) == 0, count > 1 else {
            throw PreparedCopyDescriptor.failure()
        }
        var bytes = [UInt8](repeating: 0, count: count)
        guard bytes.withUnsafeMutableBytes({ sysctlbyname("kern.osversion", $0.baseAddress, &count, nil, 0) }) == 0
        else { throw PreparedCopyDescriptor.failure() }
        return String(decoding: bytes.prefix(while: { $0 != 0 }), as: UTF8.self)
    }
}
