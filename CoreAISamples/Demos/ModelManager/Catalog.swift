import Foundation

struct CatalogModel: Sendable {
    let id: String
    let name: String
    let repo: String
    let revision: String
    let sizeBytes: Int64
    let memory: String
    let platforms: String
}

enum Catalog {
    static let models: [CatalogModel] = []
}
