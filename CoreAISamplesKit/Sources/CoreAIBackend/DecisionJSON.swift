import Foundation

extension DecisionRequest {
    public init(json: Data) throws {
        var parser = DecisionJSON(data: json)
        let root = try parser.parse()
        guard case .object(let fields) = root, case .string(let state) = fields.value("state"),
            let questions = fields.value("questions")
        else { throw DecisionError.invalidRequest("The request needs a state string and questions.") }
        self.state = state
        switch questions {
        case .object(let entries):
            self.questions = try entries.map { try Self.question(id: $0.0, value: $0.1) }
        case .array(let entries):
            self.questions = try entries.map {
                guard case .object(let fields) = $0, case .string(let id) = fields.value("id") else {
                    throw DecisionError.invalidRequest("Every question needs an ID.")
                }
                return try Self.question(id: id, value: $0)
            }
        default:
            throw DecisionError.invalidRequest("Questions must be an ordered object or array.")
        }
    }

    public func jsonData() -> Data {
        let quote = DecisionRenderer.jsonString
        let fields = questions.map { question in
            let criteria: String
            switch question.criteria {
            case .choice(let options):
                criteria =
                    "{"
                    + options.map { quote($0.id) + ":" + ($0.description.map(quote) ?? "null") }.joined(separator: ",")
                    + "}"
            case .noul(let options):
                criteria =
                    "{"
                    + options.keys.sorted().map { quote($0) + ":" + ((options[$0] ?? nil).map(quote) ?? "null") }
                    .joined(separator: ",") + "}"
            case .score(let options):
                criteria = "[" + options.map { $0.map(quote) ?? "null" }.joined(separator: ",") + "]"
            }
            let instructions = question.instructions.map { ",\"instructions\":" + quote($0) } ?? ""
            return quote(question.id) + ":{\"type\":" + quote(question.criteria.name) + instructions + ",\"criteria\":"
                + criteria + "}"
        }
        return Data(("{\"state\":" + quote(state) + ",\"questions\":{" + fields.joined(separator: ",") + "}}").utf8)
    }

    private static func question(id: String, value: DecisionJSON.Value) throws -> DecisionQuestion {
        guard case .object(let fields) = value, case .string(let type) = fields.value("type") else {
            throw DecisionError.invalidRequest("Question \(id) needs a supported type.")
        }
        let instructions: String?
        switch fields.value("instructions") {
        case .string(let text): instructions = text
        case nil, .null: instructions = nil
        default: throw DecisionError.invalidRequest("Instructions must be a string or null.")
        }
        let value = fields.value("criteria")
        let criteria: DecisionQuestion.Criteria
        switch (type, value) {
        case ("choice", .object(let options)):
            criteria = .choice(try options.map { DecisionOption(id: $0.0, description: try $0.1.optionalString()) })
        case ("noul", .object(let options)):
            var values: [String: String?] = [:]
            for (key, value) in options {
                guard key == "true" || key == "false" else {
                    throw DecisionError.invalidRequest("A noul criterion must be true or false.")
                }
                values.updateValue(try value.optionalString(), forKey: key)
            }
            criteria = .noul(values)
        case ("noul", nil), ("noul", .null):
            criteria = .noul()
        case ("score", .array(let levels)):
            criteria = .score(try levels.map { try $0.optionalString() })
        default:
            throw DecisionError.invalidRequest("Question \(id) has invalid criteria for \(type).")
        }
        return DecisionQuestion(id: id, instructions: instructions, criteria: criteria)
    }
}

private struct DecisionJSON {
    indirect enum Value {
        case string(String)
        case object([(String, Value)])
        case array([Value])
        case null
        case other

        func optionalString() throws -> String? {
            switch self {
            case .string(let value): return value
            case .null: return nil
            default: throw DecisionError.invalidRequest("Option descriptions must be strings or null.")
            }
        }
    }

    let data: Data
    var bytes: [UInt8] = []
    var offset = 0

    mutating func parse() throws -> Value {
        guard data.count <= 4 * 1_024 * 1_024 else {
            throw DecisionError.invalidRequest("The JSON request exceeds 4 MiB.")
        }
        bytes = Array(data)
        let value = try read(depth: 0)
        whitespace()
        guard offset == bytes.count else { throw invalid() }
        return value
    }

    mutating func read(depth: Int) throws -> Value {
        whitespace()
        guard depth < 64, offset < bytes.count else { throw invalid() }
        switch bytes[offset] {
        case 34: return .string(try string())
        case 123:
            offset += 1
            var fields: [(String, Value)] = []
            var keys: Set<[UInt8]> = []
            if take(125) { return .object(fields) }
            repeat {
                whitespace()
                let key = try string()
                guard keys.insert(Array(key.utf8)).inserted, take(58) else { throw invalid() }
                fields.append((key, try read(depth: depth + 1)))
                if take(125) { return .object(fields) }
            } while take(44)
            throw invalid()
        case 91:
            offset += 1
            var values: [Value] = []
            if take(93) { return .array(values) }
            repeat {
                values.append(try read(depth: depth + 1))
                if take(93) { return .array(values) }
            } while take(44)
            throw invalid()
        default:
            let start = offset
            while offset < bytes.count && ![9, 10, 13, 32, 44, 93, 125].contains(bytes[offset]) { offset += 1 }
            let token = Array(bytes[start..<offset])
            if token == Array("null".utf8) { return .null }
            if token == Array("true".utf8) || token == Array("false".utf8) { return .other }
            guard !token.isEmpty, (try? JSONDecoder().decode(Double.self, from: Data(token))) != nil else {
                throw invalid()
            }
            return .other
        }
    }

    mutating func string() throws -> String {
        guard offset < bytes.count, bytes[offset] == 34 else { throw invalid() }
        let start = offset
        offset += 1
        while offset < bytes.count {
            let byte = bytes[offset]
            offset += 1
            if byte == 34 {
                do { return try JSONDecoder().decode(String.self, from: Data(bytes[start..<offset])) } catch {
                    throw invalid()
                }
            }
            if byte == 92 { offset += 1 }
        }
        throw invalid()
    }

    mutating func whitespace() {
        while offset < bytes.count && [9, 10, 13, 32].contains(bytes[offset]) { offset += 1 }
    }

    mutating func take(_ byte: UInt8) -> Bool {
        whitespace()
        guard offset < bytes.count, bytes[offset] == byte else { return false }
        offset += 1
        return true
    }

    func invalid() -> DecisionError { .invalidRequest("Invalid or duplicate JSON key near byte \(offset).") }
}

extension Array where Element == (String, DecisionJSON.Value) {
    fileprivate func value(_ key: String) -> DecisionJSON.Value? {
        first { $0.0.utf8.elementsEqual(key.utf8) }?.1
    }
}
