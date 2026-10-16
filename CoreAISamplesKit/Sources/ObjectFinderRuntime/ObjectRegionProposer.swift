import CoreGraphics
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

    static func crop(_ image: CGImage, to region: ObjectRegion) -> CGImage? {
        guard region.isValid else { return nil }
        let rect = CGRect(
            x: region.x * Double(image.width), y: region.y * Double(image.height),
            width: region.width * Double(image.width), height: region.height * Double(image.height))
        return image.cropping(
            to: rect.integral.intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height)))
    }
}
