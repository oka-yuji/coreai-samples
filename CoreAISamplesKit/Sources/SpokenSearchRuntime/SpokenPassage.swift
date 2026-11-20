import Foundation
import SpokenSearchFeature

struct SpokenPassage: Sendable {
    let offset: Int
    let runs: [SpokenRun]

    var text: String { runs.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines) }

    var hit: SpokenHit? {
        let times = runs.compactMap(Self.times)
        guard let first = times.first, let last = times.last, last.upperBound >= first.lowerBound,
            text.unicodeScalars.contains(where: CharacterSet.alphanumerics.contains)
        else { return nil }
        return SpokenHit(id: offset, text: text, start: first.lowerBound, end: last.upperBound)
    }

    static func read(_ transcript: SpokenTranscript) -> [Self] {
        var passages: [Self] = []
        var runs: [SpokenRun] = []
        var offset = 0
        var position = 0
        var previousEnd: Double?

        func finish() {
            if !runs.isEmpty { passages.append(Self(offset: offset, runs: runs)) }
            runs.removeAll(keepingCapacity: true)
            offset = position
        }

        for run in transcript.runs {
            if let time = times(run) {
                if let previousEnd, time.lowerBound - previousEnd >= 1 { finish() }
                previousEnd = time.upperBound
            }
            var fragment = ""
            for scalar in run.text.unicodeScalars {
                fragment.unicodeScalars.append(scalar)
                position += scalar.utf16.count
                if isTerminator(scalar) || position - offset >= 128 {
                    runs.append(SpokenRun(text: fragment, start: run.start, end: run.end))
                    fragment = ""
                    finish()
                }
            }
            if !fragment.isEmpty { runs.append(SpokenRun(text: fragment, start: run.start, end: run.end)) }
        }
        finish()
        return passages
    }

    func halves() -> (Self, Self)? {
        if runs.count > 1 {
            let half = runs.count / 2
            let left = Array(runs[..<half])
            return (
                Self(offset: offset, runs: left),
                Self(offset: offset + left.reduce(0) { $0 + $1.text.utf16.count }, runs: Array(runs[half...]))
            )
        }
        guard let run = runs.first else { return nil }
        let scalars = run.text.unicodeScalars
        let count = scalars.count
        guard count > 1 else { return nil }
        let middle = scalars.index(scalars.startIndex, offsetBy: count / 2)
        let left = String(scalars[..<middle])
        let right = String(scalars[middle...])
        return (
            Self(offset: offset, runs: [SpokenRun(text: left, start: run.start, end: run.end)]),
            Self(offset: offset + left.utf16.count, runs: [SpokenRun(text: right, start: run.start, end: run.end)])
        )
    }

    private static func times(_ run: SpokenRun) -> ClosedRange<Double>? {
        guard let start = run.start, let end = run.end, start.isFinite, end.isFinite, start >= 0, end > start else {
            return nil
        }
        return start...end
    }

    private static func isTerminator(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x0A, 0x0D, 0x21, 0x2E, 0x3F, 0x3002, 0xFF01, 0xFF0E, 0xFF1F: true
        default: false
        }
    }
}
