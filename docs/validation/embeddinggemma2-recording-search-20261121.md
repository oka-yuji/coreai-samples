# Recording search on Mac and iPhone

On November 21, 2026, the production Japanese speech transcriber and EG2 FP16 ranked four passages from one synthetic recording on M5 Max. Searching for **傘は必要ですか** produced zero literal matches and ranked **傘を持ってお出かけください。** first. The player sought to 6.12 seconds and played to the end. The word **天気** started at 3.06 seconds in the same recording. [Recreate the audio and run the checks](../embeddinggemma2.md#search-a-recording-by-word-or-topic).

| Passage in recording | Start | End | Topic rank |
|---|---:|---:|---:|
| この電車は東京行きです。 | 0.00 s | 2.22 s | 4 |
| 本日の天気は晴れです。 | 2.22 s | 4.32 s | 3 |
| 午後から雨が降ります。 | 4.32 s | 6.12 s | 2 |
| 傘を持ってお出かけください。 | 6.12 s | 8.28 s | 1 |

## Recorded conditions

The input was Kyoko at a rate of 180, 16 kHz mono PCM16, 136,459 samples. The SHA-256 of the decoded little-endian PCM was `a539560b0bec8c32a5dd57b94d48ce6cadf3430b0bfc49c8e910846f7cc1dbbb`. The [fixed query and expected result](../../CoreAISamplesKit/Tests/SpokenSearchTests/Resources/topic-search.json) had SHA-256 `88b75dd42a697cd549ac03755a64e98eebec2efa9f3f75ec928e25b9584e3fa5`. Apple Speech supplied final text and audio time ranges. EG2 encoded the passages with the document role and the question with the query role, comparing normalized 768-component vectors by cosine similarity.

The run used macOS 27.2.1 (26B131), Xcode 27.2 (27B52), Swift 6.4.2, M5 Max and 128 GiB of unified memory. One request ran under the shared model lock with no explicit warm-up. AC power, low-power mode off and nominal thermal state were observed immediately before and after the diagnostic. The test exercised the existing FP16 S128/vision70 bundle, upstream revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, artifact revision `daab6dbfbae1610a87d9f181458419e3ea5ce682`, manifest SHA-256 `1a1122d998d8af9ce83c789771fe7f5f6946e181ca87572104d0a95892e65c04`, and implementation `e9e936c9a2a5829533951cdaa35221b230d30e72`.

## Normal app and verification scope

The macOS Release app used the normal sidebar, Finder picker, language controls and a verified copy of the same FP16 asset in an isolated Models library. Switching from Word to Topic and back preserved the selected recording. Pressing Play on the first topic result marked only the 6.12-second row as playing. The original model files and ledger kept their hashes. For the earlier two-occurrence word fixture, both Word results played from their individual positions, and the related recording produced no literal match.

The physical-device iOS Release build was signed. Natural recordings and listening on a physical iPhone remain owner checks. The single functional fixture covers this query, recording and FP16 configuration. Broader retrieval accuracy, PTQ recording search, speed and memory measurements remain open.
