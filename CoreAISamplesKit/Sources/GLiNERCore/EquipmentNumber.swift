import Foundation

struct EquipmentNumberWords: Decodable, Equatable, Sendable {

    var before: [String]

    var after: [String]

    var numerals: [String]

    static let bundled: EquipmentNumberWords = {
        do {
            guard let url = Bundle.module.url(forResource: "equipment_numbers", withExtension: "json") else {
                throw CocoaError(.fileNoSuchFile)
            }
            return try JSONDecoder().decode(EquipmentNumberWords.self, from: Data(contentsOf: url))
        } catch {
            preconditionFailure("The bundled equipment_numbers.json cannot be read: \(error)")
        }
    }()

    func numeralRuns(in scalars: [Unicode.Scalar]) -> [(range: Range<Int>, marked: Bool)] {
        let words = numerals.map { Array($0.unicodeScalars) }
        var runs: [Range<Int>] = []
        var index = 0
        while index < scalars.count {
            let width =
                scalars[index].halfWidth.isASCIIDigit
                ? 1 : words.filter { scalars[index...].starts(with: $0) }.map(\.count).max() ?? 0
            if width > 0, let last = runs.last, last.upperBound == index {
                runs[runs.count - 1] = last.lowerBound..<(index + width)
            } else if width > 0 {
                runs.append(index..<(index + width))
            }
            index += max(width, 1)
        }
        let folded = scalars.map(\.folded)
        let before = self.before.map { $0.unicodeScalars.map(\.folded) }
        let after = self.after.map { $0.unicodeScalars.map(\.folded) }
        return runs.map { run in
            let marked =
                before.contains { folded[..<run.lowerBound].suffix($0.count).elementsEqual($0) }
                || after.contains { folded[run.upperBound...].starts(with: $0) }
            return (run, marked)
        }
    }
}

extension Unicode.Scalar {

    fileprivate var folded: Unicode.Scalar {
        halfWidth.properties.lowercaseMapping.unicodeScalars.first ?? halfWidth
    }
}
