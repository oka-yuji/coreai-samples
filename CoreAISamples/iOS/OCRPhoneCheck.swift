#if os(iOS)
    import CoreAI
    import Darwin
    import Foundation
    import OCRCore
    import OCRRuntime
    import UIKit
    import os

    enum OCRPhoneCheck {
        struct Memory: Encodable {
            let availableBytes: UInt64
            let footprintBytes: UInt64?
            let peakFootprintBytes: UInt64?
            let thermalState: Int
        }

        struct Asset: Encodable {
            let name: String
            let cached: Bool
            let seconds: Double
        }

        struct Load: Encodable {
            let seconds: Double
            let tokenizerSeconds: Double
            let assets: [Asset]
            let before: Memory
            let after: Memory
        }

        struct Request: Encodable {
            let name: String
            let prompt: String
            let crop: [Int]?
            let resize: Int?
            let imagePx: [Int]
            let processorPx: [Int]
            let grid: [Int]
            let windows: Int
            let downscaled: Bool
            let promptTokens: Int
            let outputTokens: Int
            let finishedBy: String
            let text: String
            let rawText: String
            let tokens: [Int32]
            let visionSeconds: Double
            let firstTokenSeconds: Double
            let generateSeconds: Double
            let before: Memory
            let after: Memory
        }

        struct Device: Encodable {
            let hardware: String
            let osVersion: String
            let osBuild: String
            let architecture: String
            let lowPowerMode: Bool
            let batteryState: Int
            let batteryLevel: Float
        }

        struct Report: Encodable {
            let bundle: String
            let image: String
            let started: Date
            let device: Device
            var load: Load?
            var requests: [Request] = []
            var error: String?
            var finished: Date?
        }

        private struct Case {
            let name: String
            let prompt: String
            let instruction: String
            let crop: OCRBox?
            let resize: Int?
        }

        private static let cases = [
            Case(name: "layout-1036", prompt: "layout", instruction: OCRPrompt.layout, crop: nil, resize: 1_036),
            Case(
                name: "block02-448", prompt: "text", instruction: OCRPrompt.text,
                crop: OCRBox(x0: 83, y0: 249, x1: 526, y1: 697), resize: nil),
        ]

        static func run(bundle name: String, image file: String = "F4.png") async -> Int32 {
            let documents = URL.documentsDirectory
            let stamp = Date()
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = TimeZone(secondsFromGMT: 0)
            formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
            let folder = documents.appending(path: "OCRChecks/\(formatter.string(from: stamp))")
            var report = Report(bundle: name, image: file, started: stamp, device: await device())
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            func save() throws {
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                try encoder.encode(report).write(to: folder.appending(path: "report.json"))
            }
            do {
                try save()
                let source = try OCRImage(data: Data(contentsOf: documents.appending(path: file)))
                let bundle = documents.appending(path: name)
                let before = memory()
                let clock = ContinuousClock()
                let start = clock.now
                let runtime = try await OCRRuntime.load(bundle: bundle)
                let stats = runtime.loadStats
                report.load = Load(
                    seconds: (clock.now - start) / .seconds(1), tokenizerSeconds: stats.tokenizerSeconds,
                    assets: stats.assets.map { Asset(name: $0.name, cached: $0.cached, seconds: $0.seconds) },
                    before: before, after: memory())
                try save()
                for item in cases {
                    report.requests.append(try await recognize(item, source: source, runtime: runtime))
                    try save()
                }
            } catch {
                report.error = String(describing: error)
            }
            report.finished = Date()
            do {
                try save()
            } catch {
                print("OCR check could not write \(folder.path): \(error)")
                return 1
            }
            print("OCR check \(report.error == nil ? "completed" : "failed"): \(folder.path)")
            return report.error == nil ? 0 : 1
        }

        private static func recognize(_ item: Case, source: OCRImage, runtime: OCRRuntime) async throws -> Request {
            var image = source
            if let crop = item.crop { image = try image.cropped(to: crop) }
            if let resize = item.resize {
                image = try image.resized(width: resize, height: resize, resampling: .pillow)
            }
            let processor = try OCRImage.smartResize(width: image.width, height: image.height)
            let before = memory()
            let result = try await runtime.recognize(image, instruction: item.instruction)
            return Request(
                name: item.name, prompt: item.prompt,
                crop: item.crop.map { [$0.x0, $0.y0, $0.x1, $0.y1] }, resize: item.resize,
                imagePx: [image.width, image.height], processorPx: [processor.width, processor.height],
                grid: [result.gridHeight, result.gridWidth], windows: result.windowCount,
                downscaled: result.gridHeight * OCRVisionInput.patchSize != processor.height
                    || result.gridWidth * OCRVisionInput.patchSize != processor.width,
                promptTokens: result.promptTokens, outputTokens: result.tokens.count,
                finishedBy: result.finish == .stopToken ? "eos" : "max_new_tokens", text: result.text,
                rawText: result.rawText, tokens: result.tokens, visionSeconds: result.visionSeconds,
                firstTokenSeconds: result.firstTokenSeconds, generateSeconds: result.totalSeconds, before: before,
                after: memory())
        }

        private static func memory() -> Memory {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let result = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                }
            }
            return Memory(
                availableBytes: UInt64(os_proc_available_memory()),
                footprintBytes: result == KERN_SUCCESS ? info.phys_footprint : nil,
                peakFootprintBytes: result == KERN_SUCCESS ? UInt64(clamping: info.ledger_phys_footprint_peak) : nil,
                thermalState: ProcessInfo.processInfo.thermalState.rawValue)
        }

        @MainActor private static func device() -> Device {
            var value = utsname()
            let named = uname(&value) == 0
            let hardware = withUnsafeBytes(of: &value.machine) {
                String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self)
            }
            UIDevice.current.isBatteryMonitoringEnabled = true
            return Device(
                hardware: named ? hardware : "unknown", osVersion: UIDevice.current.systemVersion,
                osBuild: ProcessInfo.processInfo.operatingSystemVersionString,
                architecture: AIModel.deviceArchitectureName,
                lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                batteryState: UIDevice.current.batteryState.rawValue, batteryLevel: UIDevice.current.batteryLevel)
        }
    }
#endif
