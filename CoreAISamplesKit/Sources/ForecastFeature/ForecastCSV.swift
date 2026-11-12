import Foundation

enum ForecastCSV {
    private enum State { case start, plain, quoted, closed }

    static func parse(_ data: Data) throws -> [[String]] {
        guard !data.isEmpty, data.count <= 2 * 1024 * 1024, let text = String(data: data, encoding: .utf8) else {
            throw ForecastFailure.invalidData
        }
        let bytes = Array(text.utf8)
        var rows: [[String]] = []
        var row: [String] = []
        var field: [UInt8] = []
        var state = State.start
        var index = 0
        var endedRow = false
        func finishField() {
            row.append(String(decoding: field, as: UTF8.self))
            field.removeAll(keepingCapacity: true)
            state = .start
        }
        while index < bytes.count {
            let byte = bytes[index]
            endedRow = false
            if state == .quoted {
                if byte == 34 { state = .closed } else { field.append(byte) }
            } else if state == .closed && byte == 34 {
                field.append(34)
                state = .quoted
            } else if byte == 44 {
                finishField()
            } else if byte == 10 || byte == 13 {
                if byte == 13 {
                    guard index + 1 < bytes.count, bytes[index + 1] == 10 else {
                        throw ForecastFailure.invalidData
                    }
                    index += 1
                }
                finishField()
                rows.append(row)
                row.removeAll(keepingCapacity: true)
                endedRow = true
            } else if byte == 34 && state == .start {
                state = .quoted
            } else {
                guard state != .closed, byte != 34, byte != 0 else { throw ForecastFailure.invalidData }
                field.append(byte)
                state = .plain
            }
            index += 1
        }
        guard state != .quoted else { throw ForecastFailure.invalidData }
        if !endedRow {
            finishField()
            rows.append(row)
        }
        return rows
    }
}
