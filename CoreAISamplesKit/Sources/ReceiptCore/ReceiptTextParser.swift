import Foundation

public struct ReceiptTextLine: Equatable, Sendable {
    public let text: String
    public let confidence: Float

    public init(_ text: String, confidence: Float = 1) {
        self.text = text
        self.confidence = confidence
    }
}

public struct ReceiptExtractionRules: Codable, Equatable, Sendable {
    public let totalLabels: [String]
    public let currencyMarkers: [String]
    public let merchantExclusions: [String]
    public let merchantExcludedPrefixes: [String]
    public let datePattern: String
    public let foreignCurrencyPattern: String

    public init(
        totalLabels: [String], currencyMarkers: [String], merchantExclusions: [String],
        merchantExcludedPrefixes: [String], datePattern: String, foreignCurrencyPattern: String
    ) {
        self.totalLabels = totalLabels
        self.currencyMarkers = currencyMarkers
        self.merchantExclusions = merchantExclusions
        self.merchantExcludedPrefixes = merchantExcludedPrefixes
        self.datePattern = datePattern
        self.foreignCurrencyPattern = foreignCurrencyPattern
    }
}

public struct ReceiptTextParser: Sendable {
    private let rules: ReceiptExtractionRules
    private let dates: NSRegularExpression
    private let foreignCurrency: NSRegularExpression

    public init(rules: ReceiptExtractionRules) throws {
        guard !rules.totalLabels.isEmpty, rules.totalLabels.allSatisfy({ !Self.clean($0).isEmpty }),
            !rules.datePattern.isEmpty, !rules.foreignCurrencyPattern.isEmpty
        else { throw ReceiptValidationError.invalidRules }
        do {
            dates = try NSRegularExpression(pattern: rules.datePattern)
            foreignCurrency = try NSRegularExpression(pattern: rules.foreignCurrencyPattern)
        } catch { throw ReceiptValidationError.invalidRules }
        guard dates.numberOfCaptureGroups >= 3 else { throw ReceiptValidationError.invalidRules }
        self.rules = rules
    }

    public func extract(_ input: [ReceiptTextLine]) -> ReceiptExtraction {
        let lines = input.flatMap { line in
            line.text.components(separatedBy: .newlines).map {
                ReceiptTextLine(Self.clean($0), confidence: line.confidence)
            }
        }.filter { !$0.text.isEmpty }
        var issues: [ReceiptIssue] = []
        let hasForeignCurrency = lines.contains { matches(foreignCurrency, $0.text).count > 0 }
        if hasForeignCurrency { issues.append(.unsupportedCurrency) }
        let merchant = lines.first(where: { isMerchant($0.text) })?.text ?? ""
        if merchant.isEmpty { issues.append(.missingMerchant) }
        let foundDates = Set(lines.flatMap { dateValues($0.text) })
        let date = foundDates.count == 1 ? foundDates.first?.iso8601 ?? "" : ""
        if foundDates.isEmpty { issues.append(.missingDate) }
        if foundDates.count > 1 { issues.append(.ambiguousDate) }
        var amounts: Set<ReceiptAmount> = []
        var unreadable = false
        for index in lines.indices {
            guard let value = totalValue(lines[index].text) else { continue }
            let candidate = value.isEmpty && index + 1 < lines.count ? lines[index + 1].text : value
            do {
                amounts.insert(try ReceiptAmount(text: stripCurrency(candidate)))
            } catch { unreadable = true }
        }
        if amounts.isEmpty { issues.append(.missingAmount) }
        if amounts.count > 1 { issues.append(.ambiguousAmount) }
        if unreadable { issues.append(.unreadableAmount) }
        if lines.contains(where: { !$0.confidence.isFinite || $0.confidence < 0.8 || $0.confidence > 1 }) {
            issues.append(.lowRecognitionConfidence)
        }
        let amount = !hasForeignCurrency && amounts.count == 1 ? amounts.first.map { String($0.yen) } ?? "" : ""
        return ReceiptExtraction(fields: ReceiptFields(merchant: merchant, date: date, amount: amount), issues: issues)
    }

    private static func clean(_ text: String) -> String {
        text.folding(options: .widthInsensitive, locale: Locale(identifier: "en_US_POSIX"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func matches(_ expression: NSRegularExpression, _ text: String) -> [NSTextCheckingResult] {
        expression.matches(in: text, range: NSRange(text.startIndex..., in: text))
    }

    private func dateValues(_ text: String) -> [ReceiptDate] {
        matches(dates, text).compactMap { match in
            guard let span = Range(match.range, in: text),
                span.lowerBound == text.startIndex || !text[text.index(before: span.lowerBound)].isNumber,
                span.upperBound == text.endIndex || !text[span.upperBound].isNumber
            else { return nil }
            let values = (1...3).compactMap { index -> Int? in
                guard let range = Range(match.range(at: index), in: text) else { return nil }
                return Int(text[range])
            }
            guard values.count == 3 else { return nil }
            return try? ReceiptDate(year: values[0], month: values[1], day: values[2])
        }
    }

    private func totalValue(_ text: String) -> String? {
        let labels = rules.totalLabels.map(Self.clean).sorted { $0.count > $1.count }
        for label in labels where text.lowercased().hasPrefix(label.lowercased()) {
            let remainder = String(text.dropFirst(label.count))
            if let first = remainder.first, first.isLetter { continue }
            return remainder.trimmingCharacters(in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: ":")))
        }
        return nil
    }

    private func stripCurrency(_ raw: String) -> String {
        var text = Self.clean(raw)
        for marker in rules.currencyMarkers.map(Self.clean).sorted(by: { $0.count > $1.count }) where !marker.isEmpty {
            if text.lowercased().hasPrefix(marker.lowercased()) {
                text = Self.clean(String(text.dropFirst(marker.count)))
            }
            if text.lowercased().hasSuffix(marker.lowercased()) {
                text = Self.clean(String(text.dropLast(marker.count)))
            }
        }
        return text
    }

    private func isMerchant(_ text: String) -> Bool {
        guard text.unicodeScalars.contains(where: { CharacterSet.letters.contains($0) }),
            matches(dates, text).isEmpty, totalValue(text) == nil
        else { return false }
        let lower = text.lowercased()
        return !rules.merchantExclusions.contains { Self.clean($0).lowercased() == lower }
            && !rules.merchantExcludedPrefixes.contains { lower.hasPrefix(Self.clean($0).lowercased()) }
    }
}
