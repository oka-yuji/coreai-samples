import Synchronization

@testable import SpeechRuntime

final class VoxCPMPatchRecorder: Sendable {
    private let records = Mutex<[VoxCPMPatchRecord]>([])
    private let replacements: [[Float]]

    init(replacements: [[Float]] = []) {
        self.replacements = replacements
    }

    var patches: [VoxCPMPatchRecord] { records.withLock { $0 } }

    func observe(_ record: VoxCPMPatchRecord) -> [Float]? {
        records.withLock { $0.append(record) }
        return record.index < replacements.count ? replacements[record.index] : nil
    }
}
