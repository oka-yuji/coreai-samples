struct FunctionLoad: Equatable {
    let requestedCompute: String
    let expectFrequentReshapes: Bool
    let cached: Bool
    let cacheQueryError: String?
    let specializeS: Double
    let loadS: Double
}
