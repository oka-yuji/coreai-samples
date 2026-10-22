# Qwen Image 2.1 Turbo · macOS FP32

[トップ](../../README.ja.md) · [English](qwen-image21.md)

画像の生成・編集

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](../benchmarking.ja.md) · [元の記録](records/qwen-image21.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | — | このOSのネイティブデモは未提供です。 | — |
| ios | — | このOSのネイティブデモは未提供です。 | — |

### 性能と品質

| 指標 | 結果 | 集計・件数 | 条件と根拠 |
|---|---|---|---|
| バンドル・資産容量 | 62,370,841,163 B | single; n=1 | [E1](../research/qwen-image21-20261018/README.md) · bundle.jsonを除く論理ファイル容量。RAMや特殊化後容量ではない。 |
| 特殊化キャッシュ容量 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 初回ロード（特殊化を含む） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 特殊化済みキャッシュからのロード | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 要求・パイプライン完了時間 | 163.63 s | single; n=1 | [E1](../research/qwen-image21-20261018/README.md) · 生成1例。部品ロード123.42秒を含む。通常利用時の待ち時間ではない。 |
| 要求・パイプライン完了時間 | 207.24 s | single; n=1 | [E1](../research/qwen-image21-20261018/README.md) · 編集1例。ロード148.77秒を含む。同じ診断区間。 |
| 要求・パイプライン完了時間 | 174.882546 s | single; n=1 | [E2](../research/qwen-image21-off-on-20261021.json) · 診断off、既存cacheありの新process1回。warm・百分位・診断overheadの因果的な比較ではない。 |
| 要求・パイプライン完了時間 | 193.910863 s | single; n=1 | [E3](../research/qwen-image21-off-on-20261021.json) · 診断trace、既存cacheありの新process1回。warm・百分位・診断overheadの因果的な比較ではない。 |
| ピークRSS | 110,662,795,264 B | single; n=1 | [E1](../research/qwen-image21-20261018/README.md) · 参照・診断を含む検証プロセス全体の生涯ピーク。アプリ単独のRAMではない。 |
| ピークphysical footprint | 7,751,065,968 B | single; n=1 | [E1](../research/qwen-image21-20261018/README.md) · 同じ検証プロセス。RSSとは別の指標で加算しない。 |
| 推論中の常駐メモリ | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 解放後メモリの基準との差 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 課題達成率 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 参照との誤差（定義は根拠参照） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 停止受付から完了までの時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 画像と文章のCLIP score | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 人による品質評点（採点基準を固定） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 合格（記載範囲） | 固定FP32の生成・編集2例とbundle単体再実行。一般画質やモバイルは未検証。 | [E1](../research/qwen-image21-20261018/README.md) |
| 用途の品質 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 連続利用・資源解放 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 停止・再試行・回復 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](../research/qwen-image21-20261018/README.md) | 2026-10-18 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 512×512 RGBA、batch1、CFG1、8sigma。生成と参照画像1枚の編集。Python診断host、段階ロードとtensor取得を含む。 · variant=FP32; cache off; backend=Core AI; GPU preferred; route=Python conversion / diagnostic host; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; batch_size=1; concurrency=1; 未記録: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E2](../research/qwen-image21-off-on-20261021.json) | 2026-10-21 / standard / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 固定e01編集、seed20261018、512×512 RGBA、batch1、CFG1、保存8sigma。hash/prepareからPNG保存まで、段階ロード込み。診断modeごとに新process1回。cache hit自体は独立に観測していない。 · variant=FP32; no prefix KV; diagnostics off; backend=Core AI; GPU preferred; route=Python Core AI; separate process; pre-existing cache entries; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=43306970b99dda7717f7e8500179204fb0d0a9503c5836144b15b6a1d48b5852; code_revision=0a3480f617240ff43911c0a3f9a82b12cac1b8fa; batch_size=1; concurrency=1; warmups=0; power=AC; thermal=nominal |
| [E3](../research/qwen-image21-off-on-20261021.json) | 2026-10-21 / standard / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 固定e01編集、seed20261018、512×512 RGBA、batch1、CFG1、保存8sigma。hash/prepareからPNG保存まで、段階ロード込み。診断modeごとに新process1回。cache hit自体は独立に観測していない。 · variant=FP32; no prefix KV; diagnostics trace; backend=Core AI; GPU preferred; route=Python Core AI; separate process; pre-existing cache entries; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=43306970b99dda7717f7e8500179204fb0d0a9503c5836144b15b6a1d48b5852; code_revision=0a3480f617240ff43911c0a3f9a82b12cac1b8fa; batch_size=1; concurrency=1; warmups=0; power=AC; thermal=nominal |

### 導入・制限・進捗

- [変換手順（英語）](../../Tools/Conversion/recipes/qwen_image21/README.md)

- [モバイル計画 #222](https://github.com/oka-yuji/coreai-samples/issues/222)
<!-- model-report:end -->
