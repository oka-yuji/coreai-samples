struct FunctionSignature: Equatable {
    struct Input: Equatable {
        let name: String
        let typeName: String
    }

    let name: String
    let inputs: [Input]
    let stateNames: [String]

    static func find(_ name: String, in functions: [FunctionSignature]) throws -> FunctionSignature {
        guard let found = functions.first(where: { $0.name == name }) else {
            throw FunctionError.noSuchFunction(name, available: functions.map(\.name))
        }
        return found
    }

    func check(_ list: [FunctionInput]) throws {
        guard stateNames.isEmpty else { throw FunctionError.hasStates(name, stateNames) }
        var listed: Set<String> = []
        for item in list where !listed.insert(item.name).inserted { throw FunctionError.duplicateInput(item.name) }
        let types = Dictionary(inputs.map { ($0.name, $0.typeName) }, uniquingKeysWith: { first, _ in first })
        let pairs = try list.map { item -> (FunctionInput, String) in
            guard let typeName = types[item.name] else { throw FunctionError.unknownInput(item.name) }
            return (item, typeName)
        }
        if let missing = inputs.first(where: { !listed.contains($0.name) }) {
            throw FunctionError.missingInput(missing.name)
        }
        for (item, typeName) in pairs { try Self.check(item, against: typeName) }
    }

    private static func check(_ item: FunctionInput, against typeName: String) throws {
        guard let type = FunctionTensorType(typeName: typeName) else {
            throw FunctionError.unreadableType(input: item.name, typeName: typeName)
        }
        guard FunctionDtype(assetName: type.dtype) == item.dtype else {
            throw FunctionError.dtypeMismatch(input: item.name, asset: type.dtype, given: item.dtype)
        }
        guard type.dimensions.count == item.shape.count else {
            throw FunctionError.rankMismatch(input: item.name, asset: type.dimensions.count, given: item.shape.count)
        }
        for (axis, (asset, given)) in zip(type.dimensions, item.shape).enumerated() {
            guard asset < 0 ? given >= 1 : given == asset else {
                throw FunctionError.dimensionMismatch(input: item.name, axis: axis, asset: asset, given: given)
            }
        }
    }
}
