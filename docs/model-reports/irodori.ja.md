# Irodori · FP32

[トップ](../../README.ja.md) · [English](irodori.md)

音声の生成・再生

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](../benchmarking.ja.md) · [元の記録](records/irodori.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | △ | ランタイム/UIは実装済み。実アプリのGPU生成・試聴の受入は未完了。 | [E1](../irodori/demo.md) |
| ios | △ | iPhone 17 Proの診断で参照ファイル付き3生成・WAV保存・反復一致を確認。録音/ファイルUIは実装済み。通常画面の操作と試聴の本人受入は未完了。 | [E2](../irodori/iphone.md) · [E3](../irodori/reference-input-device.json) |

### 性能と品質

| 指標 | 結果 | 集計・件数 | 条件と根拠 |
|---|---|---|---|
| バンドル・資産容量 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 特殊化キャッシュ容量 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 初回ロード（特殊化を含む） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 特殊化済みキャッシュからのロード | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 要求・パイプライン完了時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| ピークRSS | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| ピークphysical footprint | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 推論中の常駐メモリ | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 解放後メモリの基準との差 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 課題達成率 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 参照との誤差（定義は根拠参照） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 停止受付から完了までの時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 再生可能な最初の音声までの時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| RTF（生成時間／音声長） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 文字誤り率CER | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 人による品質評点（採点基準を固定） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 用途の品質 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 連続利用・資源解放 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 停止・再試行・回復 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](../irodori/demo.md) | 2026-10-19 / legacy / plan | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | ネイティブ実装とCPU・ビルド検証。実アプリ生成と試聴は確認待ち。 状況の確認日は2026-10-19。モデル実行日ではない。 · variant=FP32; backend=Core AI; GPU preferred; route=plan; batch_size=1; concurrency=1; 未記録: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E2](../irodori/iphone.md) | 2026-10-19 / legacy / plan | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | ネイティブ実装とCPU・ビルド検証。実アプリ生成と試聴は確認待ち。 状況の確認日は2026-10-19。モデル実行日ではない。 · variant=FP32; backend=Core AI; GPU preferred; route=plan; batch_size=1; concurrency=1; 未記録: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E3](../irodori/reference-input-device.json) | 2026-10-29 / legacy / diagnostic | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | 合成音声3.88秒を参照し、挨拶→案内→挨拶をseed 1708で生成。WAV3本と挨拶の反復一致を確認。通常画面の操作・聴取受入とは区別する。 · variant=Reference voice FP32 v2; backend=Core AI; GPU preferred; audio/speaker/duration/decoder EFR false; route=SpeechPhone explicit reference-file diagnostic; no playback or microphone capture; source_revision=ccc78f5d480b6e51b69b2d5042a14c4da04fea6e; code_revision=PR370 working tree based on a755889e87f027497d979d41e4469805b955faf6; batch_size=1; concurrency=1; warmups=0; power=USB connected, charging; thermal=nominal (0); 未記録: artifact_revision, plan_sha256 |

### 導入・制限・進捗

- [Mac（英語）](../irodori/demo.md)

- [iPhone（英語）](../irodori/iphone.md)

- [#93](https://github.com/oka-yuji/coreai-samples/issues/93)

- [Modelsの検証（英語）](../irodori/managed-validation.md)

- [#329](https://github.com/oka-yuji/coreai-samples/issues/329)

- [参照音声ランタイム（英語）](../irodori/reference-runtime.md)
<!-- model-report:end -->
