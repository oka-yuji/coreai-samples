# Receipt Desk

Receipt Desk is the Mac receipt workflow tracked by [#509](https://github.com/oka-yuji/coreai-samples/issues/509).
The first merged unit provides the data contracts below. OCR, JEV inference, file export and the Mac interface
are tracked by [#507](https://github.com/oka-yuji/coreai-samples/issues/507) and
[#508](https://github.com/oka-yuji/coreai-samples/issues/508); this page does not claim an available app demo yet.

## Data contracts

`ReceiptCore` has no model, UI or file-system dependency. Its inputs include all text, identifiers and extraction
rules; it does not read the current time, locale or user files.

- `ReceiptAmount` stores nonnegative whole yen as `Int64`. Fractions, signs, malformed comma grouping and values
  outside `Int64` are rejected. JSON uses a decimal string such as `"1234"`, never a JSON number; a generic
  integer JSON decoder can otherwise accept a fractional token after rounding. No floating-point conversion occurs.
- `ReceiptDate` stores a Gregorian calendar day and validates it without a time zone. Its serialized form is
  `YYYY-MM-DD`; invalid dates are rejected instead of normalized.
- `ReceiptTextParser` receives text lines with recognition confidence and explicit extraction rules. It recognizes
  labelled totals and calendar dates. Missing, distinct conflicting totals/dates, foreign currency and low recognition
  confidence remain visible. A merchant is a header candidate to review, not a verified legal entity.
- `ReceiptCategory` defines eight ordered choices: travel, lodging, supplies, meals, communications, books, other and
  review. These are demo expense labels, not tax treatment or a complete chart of accounts.
- `ReceiptClassification` validates an eight-value finite probability distribution. The default maximum-probability
  threshold is 0.75; a result below it becomes review. This threshold is provisional and is not a measured correctness
  probability for Japanese receipts.
- `ReceiptRecord` keeps the source text and suggested fields. Editing invalidates confirmation. Merchant/purpose edits
  also invalidate the old category. A manually selected category and a model suggestion remain distinguishable.
  Missing fields, review categories and unsupported currency cannot be confirmed. Low OCR confidence requires
  individual review instead of bulk confirmation.

## Ledger and relative export plan

`ReceiptLedger` uses only confirmed, complete records and checks integer overflow. It sorts rows by their unique
positive identifier. A row plans a relative path such as `2026/11/supplies/2026-11-06-1.png`; the path never comes
from a model-generated string or merchant name.

Each record and ledger row carries the lowercase SHA-256 of the original receipt bytes. The runtime must compare
this fingerprint with its retained original before copying; an identifier or filename alone is not a content binding.

The CSV is UTF-8 with CRLF record separators. Text fields are quoted and embedded quotes are doubled. Text that
starts with a spreadsheet formula marker after leading whitespace/control characters receives an apostrophe prefix.
The numeric amount is the exact integer yen value. Rows contain relative receipt paths and source basenames, not
absolute input paths. The runtime must create a new destination and copy the original receipt bytes there; these
pure contracts do not perform the write.

## Verification

`swift test --package-path CoreAISamplesKit --force-resolved-versions --filter ReceiptCoreTests` covers exact amounts,
decoding, Gregorian dates, extraction fixtures, probability validation, confirmation invalidation, overflow, relative
paths and CSV cells. The Japanese fixture text is test data rather than a model-quality result. No OCR or JEV model
runs in this suite.

The application work must separately validate actual OCR, classification, stop/release ordering, export failures and
the final UI. Receipt-only input cannot establish business purpose or tax eligibility. No accounting service is called.
