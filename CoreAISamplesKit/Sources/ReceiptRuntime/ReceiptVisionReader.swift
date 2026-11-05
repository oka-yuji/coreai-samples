import CoreGraphics
import Foundation
import ImageIO
import ReceiptCore
import UniformTypeIdentifiers
import Vision

actor ReceiptVisionReader {
    func read(_ data: Data, fileExtension: String) async throws -> ReceiptRecognizedDocument {
        try Task.checkCancellation()
        if fileExtension == "pdf" { return try await readPDF(data) }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) == 1,
            let identifier = CGImageSourceGetType(source), let type = UTType(identifier as String)
        else { throw ReceiptRuntimeError.unsupportedFile }
        let ext: String
        if type.conforms(to: .png) {
            ext = "png"
        } else if type.conforms(to: .jpeg) {
            ext = "jpg"
        } else if type.conforms(to: .heic) {
            ext = "heic"
        } else if type.conforms(to: .heif) {
            ext = "heif"
        } else if type.conforms(to: .tiff) {
            ext = "tiff"
        } else {
            throw ReceiptRuntimeError.unsupportedFile
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 2400,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw ReceiptRuntimeError.unsupportedFile
        }
        let lines = try await recognize(image)
        return ReceiptRecognizedDocument(
            lines: lines, previews: [try Self.preview(image)], fileExtension: ext,
            unreadablePages: lines.isEmpty ? [1] : [])
    }

    private func readPDF(_ data: Data) async throws -> ReceiptRecognizedDocument {
        guard let provider = CGDataProvider(data: data as CFData), let document = CGPDFDocument(provider),
            document.isUnlocked, (1...5).contains(document.numberOfPages)
        else { throw ReceiptRuntimeError.unsupportedPDF }
        var lines: [ReceiptTextLine] = []
        var previews: [Data] = []
        var unreadable: [Int] = []
        for number in 1...document.numberOfPages {
            try Task.checkCancellation()
            guard let page = document.page(at: number) else { throw ReceiptRuntimeError.unsupportedPDF }
            let image = try Self.rasterize(page)
            let recognized = try await recognize(image)
            if recognized.isEmpty { unreadable.append(number) }
            lines.append(contentsOf: recognized)
            previews.append(try Self.preview(image))
        }
        return ReceiptRecognizedDocument(
            lines: lines, previews: previews, fileExtension: "pdf", unreadablePages: unreadable)
    }

    private func recognize(_ image: CGImage) async throws -> [ReceiptTextLine] {
        var request = RecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.automaticallyDetectsLanguage = false
        request.recognitionLanguages = [Locale.Language(identifier: "ja"), Locale.Language(identifier: "en")]
        request.usesLanguageCorrection = false
        let observations = try await request.perform(on: image)
        try Task.checkCancellation()
        let pieces = observations.compactMap { observation -> Piece? in
            guard let candidate = observation.topCandidates(1).first, !candidate.string.isEmpty else { return nil }
            let y = (observation.topLeft.y + observation.bottomLeft.y) / 2
            let height = abs(observation.topLeft.y - observation.bottomLeft.y)
            return Piece(
                text: candidate.string, confidence: candidate.confidence, x: observation.topLeft.x, y: y, height: height
            )
        }.sorted { $0.y == $1.y ? $0.x < $1.x : $0.y > $1.y }
        var rows: [[Piece]] = []
        for piece in pieces {
            if let last = rows.last, let anchor = last.first,
                abs(anchor.y - piece.y) <= max(0.003, min(anchor.height, piece.height) / 2)
            {
                rows[rows.count - 1].append(piece)
            } else {
                rows.append([piece])
            }
        }
        return rows.map { row in
            ReceiptTextLine(
                row.sorted { $0.x < $1.x }.map(\.text).joined(separator: " "),
                confidence: row.map(\.confidence).min() ?? 0)
        }
    }

    private struct Piece {
        let text: String
        let confidence: Float
        let x: Double
        let y: Double
        let height: Double
    }

    private static func rasterize(_ page: CGPDFPage) throws -> CGImage {
        let bounds = page.getBoxRect(.mediaBox)
        guard bounds.width.isFinite, bounds.height.isFinite, bounds.width > 0, bounds.height > 0 else {
            throw ReceiptRuntimeError.unsupportedPDF
        }
        let rotated = abs(page.rotationAngle % 180) == 90
        let pageWidth = rotated ? bounds.height : bounds.width
        let pageHeight = rotated ? bounds.width : bounds.height
        let scale = min(3, 2400 / max(pageWidth, pageHeight))
        let width = max(1, Int(ceil(pageWidth * scale)))
        let height = max(1, Int(ceil(pageHeight * scale)))
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw ReceiptRuntimeError.unsupportedPDF }
        let target = CGRect(x: 0, y: 0, width: width, height: height)
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(target)
        context.scaleBy(x: scale, y: scale)
        let pageTarget = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)
        context.concatenate(page.getDrawingTransform(.mediaBox, rect: pageTarget, rotate: 0, preserveAspectRatio: true))
        context.drawPDFPage(page)
        guard let image = context.makeImage() else { throw ReceiptRuntimeError.unsupportedPDF }
        return image
    }

    private static func preview(_ image: CGImage) throws -> Data {
        let scale = min(1, 900 / Double(max(image.width, image.height)))
        let width = max(1, Int(Double(image.width) * scale))
        let height = max(1, Int(Double(image.height) * scale))
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw ReceiptRuntimeError.unsupportedFile }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let thumbnail = context.makeImage() else { throw ReceiptRuntimeError.unsupportedFile }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw ReceiptRuntimeError.unsupportedFile
        }
        CGImageDestinationAddImage(destination, thumbnail, nil)
        guard CGImageDestinationFinalize(destination) else { throw ReceiptRuntimeError.unsupportedFile }
        return data as Data
    }
}
