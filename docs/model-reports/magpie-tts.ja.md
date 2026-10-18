# Magpie TTS Multilingual v2607

[トップ](../../README.ja.md) · [English](magpie-tts.md)

音声生成

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](../benchmarking.ja.md) · [元の記録](records/magpie-tts.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | — | このOSのネイティブデモは未提供です。 | — |
| ios | — | このOSのネイティブデモは未提供です。 | — |

### 性能と品質

| 指標 | 結果 | 集計・件数 | 条件と根拠 |
|---|---|---|---|
| バンドル・資産容量 | 1,194,186,270 B | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · Core AI資産4個の合計。元のembedding・tokenizer・Python hostは別途必要。 |
| 特殊化キャッシュ容量 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 初回ロード（特殊化を含む） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 特殊化済みキャッシュからのロード | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 要求・パイプライン完了時間 | 11.5601125 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · greeting：記録した生成呼び出し。ロード・保存を除外。 |
| 要求・パイプライン完了時間 | 9.7082895 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · instruction：記録した生成呼び出し。ロード・保存を除外。 |
| 要求・パイプライン完了時間 | 9.04915046 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · reading：記録した生成呼び出し。ロード・保存を除外。 |
| ピークRSS | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| ピークphysical footprint | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 推論中の常駐メモリ | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 解放後メモリの基準との差 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 課題達成率 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 参照との誤差（定義は根拠参照） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 停止受付から完了までの時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 再生可能な最初の音声までの時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| RTF（生成時間／音声長） | 3.50600353 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · greeting：11.5601秒／音声3.29723秒から算出。ロード・保存を除外。 |
| RTF（生成時間／音声長） | 2.51868156 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · instruction：9.70829秒／音声3.85451秒から算出。ロード・保存を除外。 |
| RTF（生成時間／音声長） | 2.66927664 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) · reading：9.04915秒／音声3.39011秒から算出。ロード・保存を除外。 |
| 文字誤り率CER | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 人による品質評点（採点基準を固定） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 合格（記載範囲） | 固定入力のFP32照合は合格。試聴と公開は別項目で未完了。 | [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) |
| 用途の品質 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 連続利用・資源解放 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 停止・再試行・回復 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md) | 2026-10-17 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 未マージの変換PRの日本語3入力。Python host。ロードと計測区間は固定した報告に従う。ネイティブデモは未提供。 · variant=FP32 research; backend=Core AI; GPU preferred; route=Python Core AI conversion host; source_revision=446bff55e1fe3d3aa29c771574e7d2b0fa322990; batch_size=1; concurrency=1; 未記録: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |

### 導入・制限・進捗

- [#165](https://github.com/oka-yuji/coreai-samples/issues/165)

- [PR #166](https://github.com/oka-yuji/coreai-samples/pull/166)

- [固定した変換結果（未マージPR）](https://github.com/oka-yuji/coreai-samples/blob/dfc037a004c38dd552fe72ef99b47d6dd7a904f7/docs/validation/magpie-conversion-20261017.md)
<!-- model-report:end -->
