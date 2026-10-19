# Qwen3-TTS CustomVoice · 1.7B

[トップ](../../README.ja.md) · [English](qwen3-tts-17b.md)

音声生成

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](../benchmarking.ja.md) · [元の記録](records/qwen3-tts-17b.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | — | このOSのネイティブデモは未提供です。 | — |
| ios | — | このOSのネイティブデモは未提供です。 | — |

### 性能と品質

| 指標 | 結果 | 集計・件数 | 条件と根拠 |
|---|---|---|---|
| バンドル・資産容量 | 6,602,093,201 B | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · Core AI資産のみ。元の資産・tokenizer・hostの入力が別途必要。 |
| 特殊化キャッシュ容量 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 初回ロード（特殊化を含む） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 特殊化済みキャッシュからのロード | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 要求・パイプライン完了時間 | 11.141144 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · greeting：記録した生成呼び出し。ロード・保存を除外。 |
| 要求・パイプライン完了時間 | 10.1524363 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · instruction：記録した生成呼び出し。ロード・保存を除外。 |
| 要求・パイプライン完了時間 | 10.088553 s | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · reading：記録した生成呼び出し。ロード・保存を除外。 |
| ピークRSS | 20,477,050,880 B | max; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · 診断1プロセス・3発話の生涯ピーク。元PyTorchモデルも常駐。ネイティブアプリのRAM要件ではない。 |
| ピークphysical footprint | 9,750,459,472 B | max; n=3 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · 3地点で読んだカーネルの生涯ピークの最大値。終了までの最大値を保証せず、PyTorchも同時常駐。 |
| 推論中の常駐メモリ | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 解放後メモリの基準との差 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 課題達成率 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 参照との誤差（定義は根拠参照） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 停止受付から完了までの時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 再生可能な最初の音声までの時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| RTF（生成時間／音声長） | 2.44323333 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · greeting：11.1411秒／音声4.56秒から算出。ロード・保存を除外。 |
| RTF（生成時間／音声長） | 2.64386363 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · instruction：10.1524秒／音声3.84秒から算出。ロード・保存を除外。 |
| RTF（生成時間／音声長） | 2.74145463 ratio | single; n=1 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) · reading：10.0886秒／音声3.68秒から算出。ロード・保存を除外。 |
| 文字誤り率CER | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 人による品質評点（採点基準を固定） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 合格（記載範囲） | 固定入力のFP32照合は合格。試聴と公開は別項目で未完了。 | [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) |
| 用途の品質 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 連続利用・資源解放 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 停止・再試行・回復 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md) | 2026-10-17 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 未マージの変換PRの日本語3入力。Python host。ロードと計測区間は固定した報告に従う。ネイティブデモは未提供。 · variant=FP32 research; backend=Core AI; GPU preferred; route=Python Core AI conversion host; source_revision=0c0e3051f131929182e2c023b9537f8b1c68adfe; batch_size=1; concurrency=1; 未記録: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |

### 導入・制限・進捗

- [#169](https://github.com/oka-yuji/coreai-samples/issues/169)

- [PR #170](https://github.com/oka-yuji/coreai-samples/pull/170)

- [固定した変換結果（未マージPR）](https://github.com/oka-yuji/coreai-samples/blob/cd79318559ecfeb42580662b928d4feed415ea8e/docs/validation/qwen3-tts-17b-conversion-20261017.md)
<!-- model-report:end -->
