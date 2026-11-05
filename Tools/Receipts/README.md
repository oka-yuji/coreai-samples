# Receipt fixtures

These four synthetic receipts exercise Receipt Desk without using personal financial documents.
The image text, purpose notes and expected labels are in
`CoreAISamplesKit/Sources/ReceiptRuntime/Resources/Examples/receipts.json`.

Regenerate the PNGs on a Mac with the installed system fonts:

```sh
swiftc -swift-version 6 -parse-as-library Tools/Receipts/generate-fixtures.swift -o .local/receipt-fixtures
.local/receipt-fixtures CoreAISamplesKit/Sources/ReceiptRuntime/Resources/Examples
```

The application reads the PNGs through Vision and sends the resulting text to JEV. Expected labels are used by
the opt-in tests and are not passed to the classifier. The images are marked as synthetic. A result on these
receipts is a smoke test, not a quality or tax-compliance claim for real documents.
