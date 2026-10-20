import CoreGraphics
import CoreVideo
import Foundation
import ObjectFinderCore
import Vision

enum ObjectRegionProposer {
    static func regions(in image: CGImage) throws -> [ObjectRegion] {
        let request = VNGenerateObjectnessBasedSaliencyImageRequest()
        try VNImageRequestHandler(cgImage: image).perform([request])
        return (request.results?.first?.salientObjects ?? []).prefix(3).compactMap { object in
            let box = object.boundingBox.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
            let region = ObjectRegion(x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height)
            return region.isValid ? region : nil
        }
    }

    static func instances(in image: CGImage) throws -> [ObjectRegion] {
        let request = VNGenerateForegroundInstanceMaskRequest()
        try VNImageRequestHandler(cgImage: image).perform([request])
        guard let observation = request.results?.first else { return [] }
        let mask = observation.instanceMask
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)
        let stride = CVPixelBufferGetBytesPerRow(mask)
        guard width > 0, height > 0, let base = CVPixelBufferGetBaseAddress(mask)?.assumingMemoryBound(to: UInt8.self)
        else { return [] }
        var bounds: [UInt8: (minX: Int, minY: Int, maxX: Int, maxY: Int)] = [:]
        for row in 0..<height {
            for column in 0..<width {
                let label = base[row * stride + column]
                guard label > 0 else { continue }
                let box = bounds[label] ?? (column, row, column, row)
                bounds[label] = (min(box.minX, column), min(box.minY, row), max(box.maxX, column), max(box.maxY, row))
            }
        }
        return bounds.keys.sorted().prefix(3).compactMap { label in
            guard let box = bounds[label] else { return nil }
            let region = ObjectRegion(
                x: Double(box.minX) / Double(width), y: Double(box.minY) / Double(height),
                width: Double(box.maxX - box.minX + 1) / Double(width),
                height: Double(box.maxY - box.minY + 1) / Double(height))
            return region.isValid ? region : nil
        }
    }

    static func crop(_ image: CGImage, to region: ObjectRegion) -> CGImage? {
        guard region.isValid else { return nil }
        let rect = CGRect(
            x: region.x * Double(image.width), y: region.y * Double(image.height),
            width: region.width * Double(image.width), height: region.height * Double(image.height))
        return image.cropping(
            to: rect.integral.intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height)))
    }
}
