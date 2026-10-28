import AVFAudio
import Foundation
import SpeechCore

protocol SpeechReferenceReading: Actor {
    func read(_ url: URL, source: SpeechReferenceSource) async throws -> SpeechReferenceAudio
    func removeRecording(_ url: URL) async
}

actor SpeechReferenceReader: SpeechReferenceReading {
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
        guard seconds <= 30 else { throw SpeechReferenceError.tooLong }
        guard seconds >= 0.5 else { throw SpeechReferenceError.tooShort }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 4096) else {
            throw SpeechReferenceError.unreadable
        }
        var mono: [Float] = []
        mono.reserveCapacity(Int(file.length))
        while file.framePosition < file.length {
            try Task.checkCancellation()
            try file.read(into: buffer, frameCount: min(4096, AVAudioFrameCount(file.length - file.framePosition)))
            guard buffer.frameLength > 0, let values = buffer.floatChannelData else {
                throw SpeechReferenceError.unreadable
            }
            for frame in 0..<Int(buffer.frameLength) {
                var sum: Float = 0
                for channel in 0..<channels {
                    let sample = values[channel][frame]
                    guard sample.isFinite else { throw SpeechReferenceError.invalidSamples }
                    sum += sample / Float(channels)
                }
                guard sum.isFinite else { throw SpeechReferenceError.invalidSamples }
                mono.append(sum)
            }
        }
        try Task.checkCancellation()
        let converted = rate == 48_000 ? mono : try resample(mono, rate: rate)
        return try SpeechReferenceAudio(samples: converted, source: source)
    }

    func removeRecording(_ url: URL) {
        guard url.lastPathComponent.hasPrefix("irodori-reference-"), url.pathExtension == "wav",
            url.deletingLastPathComponent().standardizedFileURL == URL.temporaryDirectory.standardizedFileURL
        else { return }
        try? FileManager.default.removeItem(at: url)
    }

    private func resample(_ samples: [Float], rate: Double) throws -> [Float] {
        guard let source = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1),
            let destination = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1),
            let converter = AVAudioConverter(from: source, to: destination),
            let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: 4096),
            let inputValues = input.floatChannelData,
            let output = AVAudioPCMBuffer(pcmFormat: destination, frameCapacity: 4096)
        else { throw SpeechReferenceError.unreadable }
        converter.sampleRateConverterQuality = Int(AVAudioQuality.max.rawValue)
        let targetCount = min(
            SpeechReferenceAudio.maximumSamples, Int((Double(samples.count) * 48_000 / rate).rounded()))
        var cursor = 0
        var result: [Float] = []
        result.reserveCapacity(targetCount)
        while result.count < targetCount {
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
            let count = min(Int(output.frameLength), targetCount - result.count)
            result.append(contentsOf: UnsafeBufferPointer(start: values[0], count: count))
            if status == .endOfStream { break }
            guard count > 0 else { throw SpeechReferenceError.unreadable }
        }
        return result
    }
}

@concurrent
func referenceWAV(_ reference: SpeechReferenceAudio) async throws -> Data {
    try Task.checkCancellation()
    let data = try floatWAV(samples: reference.samples)
    try Task.checkCancellation()
    return data
}
