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
    static let models: [CatalogModel] = [
        CatalogModel(
            id: "qwen3-8-27b-4bit", name: "Qwen3.8 27B (4bit)", repo: "okayuji/Qwen3.8-27B-coreai-4bit",
            revision: "c034c9857455a48cea715fec19eb49b3227b2458", sizeBytes: 16_852_520_981,
            memory: "~21 GiB while loading",
            platforms: "macOS 27, Apple silicon")
    ]
}
