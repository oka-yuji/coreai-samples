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
            revision: "29eb83f8a51dc5998552ec70bfe21d5be962c319", sizeBytes: 16_852_543_842,
            memory: "~32 GiB peak RSS while loading",
            platforms: "macOS 27, Apple silicon")
    ]
}
