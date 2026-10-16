import Foundation
import GLiNERRuntime
import TruckInspectionCore

public enum InspectionCatalogLoader {
    public static func modelFolders(in candidates: [URL]) -> [URL] {
        candidates.filter { (try? GLiNERAsset(directory: $0)) != nil }
    }

    public static func load() throws -> InspectionCatalog {
        guard let url = Bundle.module.url(forResource: "truck-inspection-ja", withExtension: "json") else {
            throw InspectionError.invalidCatalog
        }
        return try InspectionCatalog.decode(Data(contentsOf: url))
    }
}
