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
            revision: "8e24aebe036eae577d4a3bd019ba728a73358318", sizeBytes: 16_852_543_780,
            memory: "~32 GiB peak RSS while loading",
            platforms: "macOS 27, Apple silicon")
    ]
}
