import Foundation
import FoundationModels
import ImageIO
import RecipeCore

@Generable
struct ObservedIngredient {
    @Guide(description: "A short English name for visible food. Use a general label if its exact type is uncertain.")
    var name: String
    @Guide(description: "An exact count only for separate visible whole items. Otherwise nil. Never estimate weight.")
    var count: Int?
}

@Generable
struct ObservedInventory {
    @Guide(description: "Visible ingredients, including food in transparent packaging. Never infer opaque contents.")
    var foods: [ObservedIngredient]
    @Guide(description: "Brief notes about food types or containers the person must check. Do not guess contents.")
    var needsCheck: [String]
}

public actor PhotoInspector {
    public init() {}

    public func inspect(_ url: URL) async throws -> FridgePhoto {
        let started = ContinuousClock.now
        let model = SystemLanguageModel.default
        guard model.isAvailable, model.capabilities.contains(.vision) else {
            throw RecipeError.invalid("Photo recognition needs Apple Intelligence with image support on this Mac.")
        }
        let data = try Self.readImage(url)
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
            let image = CGImageSourceCreateThumbnailAtIndex(
                source, 0,
                [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceThumbnailMaxPixelSize: 1_600,
                ] as CFDictionary)
        else { throw RecipeError.invalid("The selected file is not a supported photograph.") }
        try Task.checkCancellation()
        let session = LanguageModelSession(
            model: model,
            instructions:
                "Inspect the photograph, not a recipe. List only visible food. Food in clear packaging is visible. "
                + "If meat species is uncertain, use meat and note the uncertainty. If herb variety is uncertain, "
                + "use leafy herbs and note the uncertainty. Opaque containers have unknown contents. "
                + "An unlabeled bottle is unknown. Do not infer food outside the photograph, grams or liquid volume.")
        let response = try await session.respond(
            generating: ObservedInventory.self,
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 650)
        ) {
            Attachment(image)
            "Identify visible ingredients. Count eggs only if each egg is visible. List uncertainties in needsCheck."
        }
        try Task.checkCancellation()
        guard response.content.foods.count <= 32, response.content.needsCheck.count <= 32 else {
            throw RecipeError.invalid("The photo returned too many items. Try a closer photograph.")
        }
        let items = response.content.foods.compactMap { food -> PhotoIngredient? in
            let name = food.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty, name.count <= 120 else { return nil }
            let count = food.count.flatMap { (1...99).contains($0) ? $0 : nil }
            return PhotoIngredient(name: name, visibleCount: count)
        }
        let duration = started.duration(to: .now).components
        return FridgePhoto(
            image: data, filename: url.lastPathComponent, ingredients: items,
            checks: response.content.needsCheck.map { String($0.prefix(300)) },
            seconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18)
    }

    static func readImage(_ input: URL) throws -> Data {
        guard input.isFileURL else { throw RecipeError.invalid("Choose a local photograph.") }
        let url = input.resolvingSymlinksInPath()
        guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
            throw RecipeError.invalid("Choose a regular image file.")
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var data = Data()
        while let part = try handle.read(upToCount: 65_536), !part.isEmpty {
            try Task.checkCancellation()
            data.append(part)
            guard data.count <= 20_000_000 else { throw RecipeError.invalid("Choose a photograph smaller than 20 MB.") }
        }
        return data
    }
}
