import Foundation

public struct ReceiptLedgerRow: Codable, Equatable, Sendable {
    public let id: Int
    public let sourceName: String
    public let sourceSHA256: String
    public let date: ReceiptDate
    public let merchant: String
    public let purpose: String
    public let category: ReceiptCategory
    public let amount: ReceiptAmount
    public let relativePath: String
}

public struct ReceiptLedger: Equatable, Sendable {
    public let rows: [ReceiptLedgerRow]
    public let totalYen: Int64
    public let categoryTotals: [ReceiptCategory: Int64]

    public init(records: [ReceiptRecord]) throws {
        guard Set(records.map(\.id)).count == records.count else { throw ReceiptValidationError.duplicateIdentifier }
        let confirmed = records.filter(\.isConfirmed).sorted { $0.id < $1.id }
        guard !confirmed.isEmpty else { throw ReceiptValidationError.noConfirmedReceipts }
        var rows: [ReceiptLedgerRow] = []
        var total: Int64 = 0
        var categories: [ReceiptCategory: Int64] = [:]
        for record in confirmed {
            guard record.canConfirm else { throw ReceiptValidationError.invalidFields }
            let ext = record.sourceExtension.lowercased()
            guard record.id > 0, !record.sourceName.isEmpty, record.sourceName.utf8.count <= 1024,
                !record.sourceName.contains("/"), !record.sourceName.contains("\0"),
                record.sourceSHA256.utf8.count == 64,
                record.sourceSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
                ["png", "jpg", "jpeg", "heic", "heif", "tif", "tiff", "pdf"].contains(ext)
            else { throw ReceiptValidationError.invalidSource }
            let date = try ReceiptDate(iso8601: record.fields.date)
            let amount = try ReceiptAmount(text: record.fields.amount)
            let (sum, overflow) = total.addingReportingOverflow(amount.yen)
            guard !overflow else { throw ReceiptValidationError.amountOverflow }
            total = sum
            let category = record.fields.category
            let (subtotal, categoryOverflow) = (categories[category] ?? 0).addingReportingOverflow(amount.yen)
            guard !categoryOverflow else { throw ReceiptValidationError.amountOverflow }
            categories[category] = subtotal
            let path = "\(date.yearText)/\(date.monthText)/\(category.rawValue)/\(date.iso8601)-\(record.id).\(ext)"
            rows.append(
                ReceiptLedgerRow(
                    id: record.id, sourceName: record.sourceName, sourceSHA256: record.sourceSHA256, date: date,
                    merchant: record.fields.merchant,
                    purpose: record.fields.purpose, category: category, amount: amount, relativePath: path))
        }
        self.rows = rows
        totalYen = total
        categoryTotals = categories
    }

    public var csv: Data {
        let header = "receipt_id,date,merchant,purpose,category,total_jpy,receipt_path,source_name"
        let records = rows.map { row in
            [
                String(row.id), row.date.iso8601, Self.csvText(row.merchant), Self.csvText(row.purpose),
                row.category.rawValue, String(row.amount.yen), Self.csvText(row.relativePath),
                Self.csvText(row.sourceName),
            ].joined(separator: ",")
        }
        return Data(([header] + records).joined(separator: "\r\n").appending("\r\n").utf8)
    }

    private static func csvText(_ text: String) -> String {
        let significant = text.trimmingCharacters(in: .whitespacesAndNewlines.union(.controlCharacters))
        let dangerous = significant.first.map { "=+-@".contains($0) } ?? false
        let safe = dangerous ? "'" + text : text
        return "\"" + safe.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }
}
