import Foundation
import ImageGenerationCore
import Testing

struct ImageExecutionTests {
    private func budget(_ amount: UInt64 = 100) throws -> ImageMemoryBudget {
        try ImageMemoryBudget(
            profileID: "profile", manifestSHA256: String(repeating: "a", count: 64),
            components: Dictionary(
                uniqueKeysWithValues: ImageModelComponent.allCases.map {
                    ($0, try ImageComponentMemoryBudget(loadingAdditionalBytes: amount, inferenceAdditionalBytes: 20))
                }))
    }

    @Test("Loading and inference use distinct additional budgets and the larger memory reserve")
    func reserves() throws {
        let budget = try budget()
        let small = ImageMemoryObservation(
            scope: .processDirtyBudget, availableBytes: 536_871_012, footprintBytes: 0,
            physicalMemoryBytes: 8_000_000_000)
        #expect(try budget.check(.text, phase: .loading, observation: small) == 536_870_912)
        let insufficient = ImageMemoryObservation(
            scope: .processDirtyBudget, availableBytes: 536_870_932, footprintBytes: 0,
            physicalMemoryBytes: 8_000_000_000)
        #expect(try budget.check(.text, phase: .inference, observation: insufficient) == 536_870_912)
        #expect(
            throws: ImageExecutionError.insufficientMemory(
                component: .text, required: 536_871_012, available: 536_870_932)
        ) {
            try budget.check(.text, phase: .loading, observation: insufficient)
        }
        let large = ImageMemoryObservation(
            scope: .processDirtyBudget, availableBytes: 5_000_000_000, footprintBytes: 4_000_000_001,
            physicalMemoryBytes: 16_000_000_000)
        #expect(try budget.check(.vision, phase: .loading, observation: large) == 1_800_000_001)
        let mac = ImageMemoryObservation(
            scope: .estimatedSystemHeadroom, availableBytes: 5_000_000_000, footprintBytes: 4_000_000_001,
            physicalMemoryBytes: 16_000_000_000)
        #expect(try budget.check(.vision, phase: .loading, observation: mac) == 3_200_000_000)
    }

    @Test("Missing memory observations invalid pins and incomplete budgets cannot authorize a load")
    func invalidBudgets() throws {
        let budget = try budget()
        for scope in [ImageMemoryScope.processDirtyBudget, .estimatedSystemHeadroom, .unavailable] {
            let unknown = ImageMemoryObservation(
                scope: scope, availableBytes: nil, footprintBytes: 100, physicalMemoryBytes: 8_000_000_000)
            #expect(throws: ImageExecutionError.memoryUnavailable) {
                try budget.check(.text, phase: .loading, observation: unknown)
            }
        }
        #expect(throws: ImageExecutionError.budgetNotConfigured) {
            try ImageMemoryBudget(profileID: "x", manifestSHA256: String(repeating: "0", count: 64), components: [:])
        }
        #expect(throws: ImageExecutionError.budgetNotConfigured) {
            try ImageMemoryBudget(profileID: "x", manifestSHA256: "bad", components: budget.components)
        }
        #expect(throws: ImageExecutionError.budgetNotConfigured) {
            try ImageComponentMemoryBudget(loadingAdditionalBytes: 1, inferenceAdditionalBytes: 0)
        }
        var json = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(budget)) as? [String: Any])
        var components = try #require(json["components"] as? [String: [String: Any]])
        components["text"]?["loadingAdditionalBytes"] = 0
        json["components"] = components
        let corrupt = try JSONSerialization.data(withJSONObject: json)
        #expect(throws: ImageExecutionError.budgetNotConfigured) {
            try JSONDecoder().decode(ImageMemoryBudget.self, from: corrupt)
        }
    }

    @Test("Integer overflow and unavailable process limits are rejected without reducing the reserve")
    func overflowingBudget() throws {
        let budget = try budget(.max)
        let observation = ImageMemoryObservation(
            scope: .processDirtyBudget, availableBytes: 1_000_000_000, footprintBytes: 1,
            physicalMemoryBytes: 8_000_000_000)
        #expect(throws: ImageExecutionError.budgetNotConfigured) {
            try budget.check(.text, phase: .loading, observation: observation)
        }
        let overflowing = ImageMemoryObservation(
            scope: .processDirtyBudget, availableBytes: .max, footprintBytes: 1, physicalMemoryBytes: 8_000_000_000)
        #expect(throws: ImageExecutionError.memoryUnavailable) { try overflowing.budgetBytes() }
    }

    @Test("Tensor boundaries reject invalid masks nonfinite floats and mismatched physical types")
    func tensors() throws {
        let float = ImageTensorContract(name: "x", shape: [2], dtype: .float32, layout: "row-major")
        try ImageTensorData.floats([1, 2]).validate(float)
        for value in [ImageTensorData.floats([1]), .floats([.nan, 1]), .integers([1, 2])] {
            #expect(throws: ImageExecutionError.invalidTensor("x")) { try value.validate(float) }
        }
        let mask = ImageTensorContract(name: "mask", shape: [2], dtype: .bool, layout: "row-major")
        try ImageTensorData.mask([0, 1]).validate(mask)
        #expect(throws: ImageExecutionError.invalidTensor("mask")) { try ImageTensorData.mask([1, 2]).validate(mask) }
    }

    @Test(
        "Boolean validation covers every byte across whole words and remaining bytes", arguments: [1, 7, 8, 9, 16, 19])
    func maskWords(_ count: Int) throws {
        let contract = ImageTensorContract(name: "mask", shape: [count], dtype: .bool, layout: "row-major")
        let valid = (0..<count).map { UInt8($0 % 2) }
        try ImageTensorData.mask(valid).validate(contract)
        for index in 0..<count {
            var invalid = valid
            invalid[index] = 128
            #expect(throws: ImageExecutionError.invalidTensor("mask")) {
                try ImageTensorData.mask(invalid).validate(contract)
            }
        }
    }

    @Test("Requests reject oversized text invalid seeds reference sizes and noise before inference")
    func requestValidation() throws {
        let id = UUID()
        try ImageGenerationRequest(id: id, prompt: "image", seed: 0).validate()
        #expect(throws: ImageGenerationError.invalidTokens) {
            try ImageGenerationRequest(id: id, prompt: " \n", seed: 0).validate()
        }
        #expect(throws: ImageGenerationError.inputTooLarge(262_145)) {
            try ImageGenerationRequest(id: id, prompt: String(repeating: "x", count: 262_145), seed: 0).validate()
        }
        #expect(throws: ImageGenerationError.invalidSeed) {
            try ImageGenerationRequest(id: id, prompt: "image", seed: .max).validate()
        }
        #expect(throws: ImageGenerationError.invalidShape) {
            try ImageGenerationRequest(id: id, prompt: "image", seed: 1, initialNoise: [0]).validate()
        }
        let pixels = try ImageGenerationPixels(width: 1, height: 1, rgba: [0, 0, 0, 255])
        #expect(throws: ImageGenerationError.invalidShape) {
            try ImageGenerationRequest(id: id, prompt: "image", seed: 1, reference: pixels).validate()
        }
    }
}
