import CryptoKit
import Foundation

struct FunctionRecord: Codable, Equatable {
    struct Tensor: Codable, Equatable {
        let name: String
        let dtype: FunctionDtype
        let shape: [Int]
        let file: String?
        let bytes: Int
        let sha256: String

        init(_ tensor: FunctionTensor, file: String? = nil) {
            name = tensor.name
            dtype = tensor.dtype
            shape = tensor.shape
            self.file = file
            bytes = tensor.data.count
            sha256 = hex(Data(SHA256.hash(data: tensor.data)))
        }
    }

    var record = "function"
    var byteOrder = "little"
    let mainHash: String?
    let function: String
    let inputs: [Tensor]
    let outputs: [Tensor]
    let requestedCompute: String
    let expectFrequentReshapes: Bool
    let cached: Bool
    let cacheQueryError: String?
    let specializeS: Double
    let loadS: Double
    let callS: [Double]
    let repeatIdentical: Bool
    let repeatDifferingCalls: [Int]

    init(
        mainHash: String?, function: String, inputs: [Tensor], outputs: [Tensor], load: FunctionLoad, callS: [Double],
        differingCalls: [Int]
    ) {
        self.mainHash = mainHash
        self.function = function
        self.inputs = inputs
        self.outputs = outputs
        requestedCompute = load.requestedCompute
        expectFrequentReshapes = load.expectFrequentReshapes
        cached = load.cached
        cacheQueryError = load.cacheQueryError
        specializeS = load.specializeS
        loadS = load.loadS
        self.callS = callS
        repeatIdentical = differingCalls.isEmpty
        repeatDifferingCalls = differingCalls
    }
}
