# Receipt Desk

Receipt Desk is the Mac receipt workflow tracked by [#509](https://github.com/oka-yuji/coreai-samples/issues/509).
The core and runtime provide the contracts below. The Mac interface and Models integration are tracked by
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

## Runtime

`ReceiptRuntime` runs on macOS 27 or later. `ReceiptPipeline` retains at most 20 receipt snapshots, each at most
20 MiB. It accepts one receipt per JPEG, PNG, HEIC/HEIF or single-frame TIFF file, or a PDF with one to five pages.
The file reader rejects non-regular files, leaf symlinks, empty files, changed reads and duplicate content.
An imported snapshot contains the original bytes and their SHA-256; editing or deleting the source afterward does
not silently change the evidence exported with a ledger.

Vision recognizes fixed Japanese/English text locally with automatic language detection disabled. Core Graphics renders PDF pages; image previews are bounded
PNG copies. Empty pages remain explicitly unreadable. The deterministic parser proposes a merchant, date and labelled
total. Ambiguous or unrecognized values need correction; OCR confidence is not a correctness guarantee.

JEV receives the recognized text, corrected merchant and optional purchase purpose as JSON, together with one fixed
question and eight ordered choices. `TypedDecisionModel` performs actual Core AI inference. Neither the expected
sample labels nor the UI language is fed into the model. The answer can change only the proposed category; it never
supplies a total, date, folder path or accounting transaction. The first version performs direct classification and
has no Decision RAG tree or embedding index.

A pipeline has one active operation. Cancellation joins native work before another request can use it. Release
permanently closes the instance, waits for any active work, then clears the model and snapshots. A new owner must
create a new pipeline. Timings returned by read/classify describe that operation and exclude model download/preparation.
They are diagnostic timings, not a throughput promise.

Export validates every confirmed ledger row against the retained source fingerprint. It writes receipt copies,
`ledger.csv` and `ledger.json` into a private staging directory, then publishes a new `ReceiptDesk-<UUID>` directory
with an exclusive rename. Open directory descriptors and recorded file identities keep writes attached to the owned
staging tree. Publication rechecks exact directory inventories and hashes each retained output file descriptor.
Cleanup removes only recorded entries whose identities still match; replaced or unrecognized contents
are preserved and reported as a cleanup failure. A normal failure or cancellation before publication removes staging;
an existing destination is never overwritten. Successful publication is the commit point. The original receipt is never moved or modified.
JSON amounts are decimal strings; CSV amounts are exact integer yen. This is a reviewed draft ledger, not a tax filing.

## Runtime checks and samples

`swift test --package-path CoreAISamplesKit --force-resolved-versions --filter ReceiptRuntimeTests` uses fake readers
and decision engines for source limits, reentrant cancellation/release, retry, invalid classifications, source binding,
transaction failure and cancellation. Native checks are skipped by default.

The four images under `ReceiptRuntime/Resources/Examples` are synthetic demo receipts, including Japanese text and
a mixed personal/business purchase. Their generator and input manifest are documented in `Tools/Receipts/README.md`.
Expected categories are test assertions only. These four examples cannot establish receipt classification accuracy.

Run the following native checks under the shared heavy-job lock described in `AGENTS.md`:

```sh
RECEIPT_VISION_TEST=1 swift test --package-path CoreAISamplesKit -c release --filter ReceiptNativeTests.nativeOCR
RECEIPT_JEV_BUNDLE=/path/to/public-jev-bundle RECEIPT_NATIVE_REPORT=/new/report.json swift test --package-path CoreAISamplesKit -c release --filter ReceiptNativeTests.nativeJEV
```

The JEV test requires the published `okayuji/jev-27b-coreai` revision
`acc4cc835f64a75ab9ecee9d5d2e119a3bd29212`, whose `decision.json` SHA-256 is
`2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d`.
It checks actual OCR/classification and the exported CSV/original-byte hashes. Host and native results do not stand
in for the final Mac UI, user receipts, Japanese receipt quality, accounting correctness or iPhone acceptance.
Receipt-only input cannot establish business purpose or tax eligibility. No accounting service is called.

On 2026-11-06, macOS 27.2 (26B123), Xcode 27.0 and an M5 Max with 128 GiB passed the native image/PDF check
and the public JEV check. The four synthetic receipts matched the expected supplies/meals/travel/review labels.
The three confirmable receipts totaled JPY 3,320; the mixed purchase remained review. The check also compared
exported original-byte hashes and CSV bytes. This is one functional smoke run, not a receipt benchmark or UI acceptance.

The initial native OCR check exposed two issues: automatic language detection read Japanese yen suffixes as letters,
and the PDF drawing transform did not enlarge pages. Fixed language selection and an explicit rendering scale passed
the same unchanged sample assertions. The parser never converted the misread letters into money.
