import EmbeddingCore
import SpokenSearchFeature

struct SpokenIndexedPassage: Sendable {
    let hit: SpokenHit
    let vector: EmbeddingVector
}
