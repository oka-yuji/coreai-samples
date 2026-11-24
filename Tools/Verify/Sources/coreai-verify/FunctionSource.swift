protocol FunctionSource {
    func describe() throws -> (mainHash: String?, functions: [FunctionSignature])
    func load(_ function: String) async throws -> FunctionLoad
    func call(_ inputs: [FunctionTensor]) async throws -> [FunctionTensor]
}
