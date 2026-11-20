import AVFAudio
import Foundation
import SpeechCore

protocol SpeechReferenceReading: Actor {
    func read(_ url: URL, source: SpeechReferenceSource) async throws -> SpeechReferenceAudio
    func removeRecording(_ url: URL) async
}

actor SpeechReferenceReader: SpeechReferenceReading {
    let format: SpeechReferenceFormat

    init(format: SpeechReferenceFormat = .irodori) {
        self.format = format
    }

    func read(_ url: URL, source: SpeechReferenceSource) throws -> SpeechReferenceAudio {
        guard url.isFileURL else { throw SpeechReferenceError.unreadable }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        try Task.checkCancellation()
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        } catch { throw SpeechReferenceError.unreadable }
        let rate = file.processingFormat.sampleRate
        let channels = Int(file.processingFormat.channelCount)
        guard rate.isFinite, (8_000...192_000).contains(rate), (1...8).contains(channels), file.length > 0 else {
            throw SpeechReferenceError.unreadable
        }
        let seconds = Double(file.length) / rate
        guard seconds <= maximumSeconds else { throw tooLongError }
        guard seconds >= minimumSeconds else { throw tooShortError }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 4096) else {
            throw SpeechReferenceError.unreadable
        }
        var mono: [Float] = []
        mono.reserveCapacity(Int(file.length))
        let divisor = Float(channels)
        while file.framePosition < file.length {
            try Task.checkCancellation()
            try file.read(into: buffer, frameCount: min(4096, AVAudioFrameCount(file.length - file.framePosition)))
            guard buffer.frameLength > 0, let values = buffer.floatChannelData else {
                throw SpeechReferenceError.unreadable
            }
            for frame in 0..<Int(buffer.frameLength) {
                let first = values[0][frame]
                guard first.isFinite else { throw SpeechReferenceError.invalidSamples }
                guard channels > 1 else {
                    mono.append(first)
                    continue
                }
                var sum: Float = 0
                sum += first / divisor
                for channel in 1..<channels {
                    let sample = values[channel][frame]
                    guard sample.isFinite else { throw SpeechReferenceError.invalidSamples }
                    sum += sample / divisor
                }
                guard sum.isFinite else { throw SpeechReferenceError.invalidSamples }
                mono.append(sum)
            }
        }
        guard mono.count == Int(file.length) else { throw SpeechReferenceError.unreadable }
        try Task.checkCancellation()
        let converted = rate == Double(format.sampleRate) ? mono : try resample(mono, rate: rate)
        try Task.checkCancellation()
        return try SpeechReferenceAudio(samples: converted, source: source, format: format)
    }

    func removeRecording(_ url: URL) {
        guard url.lastPathComponent.hasPrefix("irodori-reference-"), url.pathExtension == "wav",
            url.deletingLastPathComponent().standardizedFileURL == URL.temporaryDirectory.standardizedFileURL
        else { return }
        try? FileManager.default.removeItem(at: url)
    }

    static func completeVoxResampling(
        _ output: inout [Float], sourceCount: Int, consumedCount: Int, sourceRate: Double
    ) throws {
        guard sourceCount > 0, consumedCount == sourceCount, sourceRate.isFinite, sourceRate > 0 else {
            throw SpeechReferenceError.unreadable
        }
        let exact = Double(sourceCount) * Double(SpeechReferenceFormat.voxcpm2.sampleRate) / sourceRate
        let required = Int(exact.rounded(.up))
        if output.count >= required {
            output.removeLast(output.count - required)
            return
        }
        guard output.count == required - 1, exact.rounded(.down) != exact else {
            throw SpeechReferenceError.unreadable
        }
        output.append(0)
    }

    private var minimumSeconds: Double {
        switch format {
        case .irodori: 0.5
        case .voxcpm2: 5
        }
    }

    private var maximumSeconds: Double {
        switch format {
        case .irodori: 30
        case .voxcpm2: 10
        }
    }

    private var tooShortError: SpeechReferenceError {
        switch format {
        case .irodori: .tooShort
        case .voxcpm2: .voxcpmTooShort
        }
    }

    private var tooLongError: SpeechReferenceError {
        switch format {
        case .irodori: .tooLong
        case .voxcpm2: .voxcpmTooLong
        }
    }

    private func targetCount(sourceCount: Int, rate: Double) -> Int {
        let exact = Double(sourceCount) * Double(format.sampleRate) / rate
        switch format {
        case .irodori: return min(SpeechReferenceAudio.maximumSamples, Int(exact.rounded()))
        case .voxcpm2: return Int(exact.rounded(.up))
        }
    }

    private func resample(_ samples: [Float], rate: Double) throws -> [Float] {
        guard let source = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1),
            let destination = AVAudioFormat(
                standardFormatWithSampleRate: Double(format.sampleRate), channels: 1),
            let converter = AVAudioConverter(from: source, to: destination),
            let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: 4096),
            let inputValues = input.floatChannelData,
            let output = AVAudioPCMBuffer(pcmFormat: destination, frameCapacity: 4096)
        else { throw SpeechReferenceError.unreadable }
        converter.sampleRateConverterQuality = Int(AVAudioQuality.max.rawValue)
        let targetCount = targetCount(sourceCount: samples.count, rate: rate)
        let drains = format == .voxcpm2
        var cursor = 0
        var result: [Float] = []
        result.reserveCapacity(targetCount)
        while drains || result.count < targetCount {
            try Task.checkCancellation()
            var conversionError: NSError?
            let status = converter.convert(to: output, error: &conversionError) { requested, inputStatus in
                if cursor == samples.count {
                    inputStatus.pointee = .endOfStream
                    return nil
                }
                let count = min(Int(requested), Int(input.frameCapacity), samples.count - cursor)
                input.frameLength = AVAudioFrameCount(count)
                samples.withUnsafeBufferPointer { values in
                    if let base = values.baseAddress {
                        inputValues[0].update(from: base.advanced(by: cursor), count: count)
                    }
                }
                cursor += count
                inputStatus.pointee = .haveData
                return input
            }
            guard conversionError == nil, status != .error, let values = output.floatChannelData else {
                throw SpeechReferenceError.unreadable
            }
            let count = min(Int(output.frameLength), max(0, targetCount - result.count))
            result.append(contentsOf: UnsafeBufferPointer(start: values[0], count: count))
            if status == .endOfStream {
                if drains {
                    try Self.completeVoxResampling(
                        &result, sourceCount: samples.count, consumedCount: cursor, sourceRate: rate)
                }
                break
            }
            guard output.frameLength > 0 else { throw SpeechReferenceError.unreadable }
        }
        return result
    }
}

@concurrent
func referenceWAV(_ reference: SpeechReferenceAudio) async throws -> Data {
    try Task.checkCancellation()
    let data = try floatWAV(reference: reference)
    try Task.checkCancellation()
    return data
}
