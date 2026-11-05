import AppKit
import Foundation

@main
struct ReceiptFixtureGenerator {
    struct Example: Decodable {
        let file: String
        let lines: [String]
    }

    enum Failure: Error { case arguments, image, path }

    @MainActor
    static func main() throws {
        guard CommandLine.arguments.count == 2 else { throw Failure.arguments }
        let directory = URL(filePath: CommandLine.arguments[1], directoryHint: .isDirectory)
        let examples = try JSONDecoder().decode(
            [Example].self, from: Data(contentsOf: directory.appending(path: "receipts.json")))
        for example in examples {
            guard example.file == URL(filePath: example.file).lastPathComponent,
                example.file.hasSuffix(".png"), example.lines.count <= 18,
                let bitmap = NSBitmapImageRep(
                    bitmapDataPlanes: nil, pixelsWide: 900, pixelsHigh: 1200, bitsPerSample: 8,
                    samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                    bytesPerRow: 3600, bitsPerPixel: 32),
                let context = NSGraphicsContext(bitmapImageRep: bitmap)
            else { throw Failure.image }
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = context
            NSColor.white.setFill()
            NSRect(x: 0, y: 0, width: 900, height: 1200).fill()
            for (index, line) in example.lines.enumerated() {
                let font: NSFont =
                    index == 0
                    ? .systemFont(ofSize: 40, weight: .bold)
                    : .monospacedSystemFont(ofSize: 30, weight: .regular)
                let text = NSAttributedString(string: line, attributes: [.font: font, .foregroundColor: NSColor.black])
                text.draw(in: NSRect(x: 70, y: 1090 - index * 64, width: 780, height: 55))
            }
            NSAttributedString(
                string: "SYNTHETIC DEMO RECEIPT",
                attributes: [
                    .font: NSFont.systemFont(ofSize: 16), .foregroundColor: NSColor.darkGray,
                ]
            ).draw(in: NSRect(x: 70, y: 75, width: 780, height: 30))
            NSGraphicsContext.restoreGraphicsState()
            guard let data = bitmap.representation(using: .png, properties: [:]) else { throw Failure.image }
            try data.write(to: directory.appending(path: example.file), options: .atomic)
            print(example.file, data.count)
        }
    }
}
