import Foundation
import TruckInspectionCore

public enum InspectionCatalogLoader {
    public static func modelFolders(in candidates: [URL]) -> [URL] {
        struct Identity: Decodable { let model: String }
        return candidates.filter { folder in
            guard let data = try? Data(contentsOf: folder.appending(path: "contract.json")),
                let identity = try? JSONDecoder().decode(Identity.self, from: data)
            else { return false }
            return identity.model.lowercased().contains("gliner")
        }
    }

    public static func load() throws -> InspectionCatalog {
        guard let url = Bundle.module.url(forResource: "truck-inspection-ja", withExtension: "json") else {
            throw InspectionError.invalidCatalog
        }
        return try InspectionCatalog.decode(Data(contentsOf: url))
    }
}
