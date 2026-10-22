# Qwen Image 2.1 Turbo

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
| 参照との誤差（定義は根拠参照） | 0.0233002635 error | max; n=16 | [E4](../research/qwen-image21-w4-smoke-20261022.json) · 2入力×8stepのDiT有効joint領域における正規化RMSEの最大値。凍結F16上限0.0009765625に不合格。正規化最大誤差にも別の基準がある。 |
| 参照との誤差（定義は根拠参照） | 1.77816848e-05 error | max; n=4 | [E6](../research/qwen-image21-selective-dit-20261022.json) · 8block候補w4-blocks16-23-f32の固定step0入力4件におけるDiT full-joint正規化RMSEの最大値（e08）。凍結F32上限0.0001で4件とも合格。60run suiteではない。 |
| 参照との誤差（定義は根拠参照） | 0.000311233135 error | max; n=4 | [E6](../research/qwen-image21-selective-dit-20261022.json) · 8block候補w4-blocks16-23-f32の固定step0入力4件におけるDiT full-joint正規化最大誤差の最大値（e08）。凍結F32上限0.001で4件とも合格。60run suiteではない。 |
| 停止受付から完了までの時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 画像と文章のCLIP score | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 人による品質評点（採点基準を固定） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 不合格（記載基準） | モバイル候補は未合格。W4/F16 DiTは固定入力（2入力×8step）で不合格。全部品のW4/F32は固定比較1,230件中32件がfull valid jointで不合格（target領域480件は合格）。8block保持DiTは固定step0の4入力で合格したが、60run suite・rollout・品質・実機は未実施。従来FP32の合格は別の証拠として保持する。 | [E4](../research/qwen-image21-w4-smoke-20261022.json) · [E5](../research/qwen-image21-w4-f32-diagnostic-20261022.json) · [E6](../research/qwen-image21-selective-dit-20261022.json) |
| 用途の品質 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 連続利用・資源解放 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 停止・再試行・回復 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](../research/qwen-image21-20261018/README.md) | 2026-10-18 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 512×512 RGBA、batch1、CFG1、8sigma。生成と参照画像1枚の編集。Python診断host、段階ロードとtensor取得を含む。 · variant=FP32; cache off; backend=Core AI; GPU preferred; route=Python conversion / diagnostic host; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; batch_size=1; concurrency=1; 未記録: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E2](../research/qwen-image21-off-on-20261021.json) | 2026-10-21 / standard / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 固定e01編集、seed20261018、512×512 RGBA、batch1、CFG1、保存8sigma。hash/prepareからPNG保存まで、段階ロード込み。診断modeごとに新process1回。cache hit自体は独立に観測していない。 · variant=FP32; no prefix KV; diagnostics off; backend=Core AI; GPU preferred; route=Python Core AI; separate process; pre-existing cache entries; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=43306970b99dda7717f7e8500179204fb0d0a9503c5836144b15b6a1d48b5852; code_revision=0a3480f617240ff43911c0a3f9a82b12cac1b8fa; batch_size=1; concurrency=1; warmups=0; power=AC; thermal=nominal |
| [E3](../research/qwen-image21-off-on-20261021.json) | 2026-10-21 / standard / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 固定e01編集、seed20261018、512×512 RGBA、batch1、CFG1、保存8sigma。hash/prepareからPNG保存まで、段階ロード込み。診断modeごとに新process1回。cache hit自体は独立に観測していない。 · variant=FP32; no prefix KV; diagnostics trace; backend=Core AI; GPU preferred; route=Python Core AI; separate process; pre-existing cache entries; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=43306970b99dda7717f7e8500179204fb0d0a9503c5836144b15b6a1d48b5852; code_revision=0a3480f617240ff43911c0a3f9a82b12cac1b8fa; batch_size=1; concurrency=1; warmups=0; power=AC; thermal=nominal |
| [E4](../research/qwen-image21-w4-smoke-20261022.json) | 2026-10-22 / legacy / diagnostic | 未記録 / 未記録 / 未記録 | 固定g01/e01、seed20261018、512正方形、batch1、CFG1、8step、joint2304、prefix KVなし。資産revisionは5部品記録のhashで検証済みbundleではない。電源/熱は未記録で速度の主張には使わない。 · variant=qmi-mobile-w4-v1; DiT F16 math; other components F32 math; backend=Core AI; GPU preferred; route=Python Core AI fixed-input and captured rollout versus original Diffusers on MPS; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=ce6e70483c6a47f456fb1a4bd92e95fe6993b31fada74e460311eaf90f2f77e4; code_revision=ec5f51096bb1a3f41d938045163bebdd5c76d14c; batch_size=1; concurrency=1; warmups=0; 未記録: device, os, toolchain, power, thermal |
| [E5](../research/qwen-image21-w4-f32-diagnostic-20261022.json) | 2026-10-22 / legacy / diagnostic | 未記録 / 未記録 / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | 凍結60run suite（512正方形、batch1、CFG1、8step、prefix KVなし）。原典Bは全60runと必須2反復を完了し、反復差は0。Core AIの固定比較は1,230件中32件がfull valid jointで不合格（e08/e09の20261018・20261021、各8step）、target領域480件は全合格。自由rolloutは60run中16runで中止。JSONに書出し資産のinventory hashはなく、電源/熱は未記録で速度の主張には使わない。 · variant=qmi-mobile-w4-f32-v1; F32 math for all components; backend=Core AI; GPU preferred; route=Python Core AI fixed-input comparisons and partial free rollout versus the independent same-weight source reference B on MPS; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; code_revision=f5cca6f12dde207fa9a9abc6c4af6eb31195951d; batch_size=1; concurrency=1; warmups=0; 未記録: device, os, artifact_revision, power, thermal |
| [E6](../research/qwen-image21-selective-dit-20261022.json) | 2026-10-22 / legacy / diagnostic | M4 Max / macOS 27.0 / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | W4/F32 cohortの固定DiT step0入力4件（g01/e01/e08/e09、20261018）を再利用。joint2304、512正方形、batch1、CFG1。0始まりのDiT block16〜23のLinear weightをF16で保持するw4-blocks16-23-f32はfull-joint/targetとも4/4合格、w4-blocks20-21-f32はfull-jointで2/4（e08/e09）不合格。資産revisionは8block DiTのmain.mlirbのSHA-256で、他の資産hashはJSONにある。60run suite・rollout・品質・実機の結果ではなくverified=false。電源/熱は未記録で速度の主張には使わない。 · variant=w4-blocks16-23-f32 and w4-blocks20-21-f32; DiT only; F32 math; listed DiT blocks keep F16 Linear weights; backend=Core AI; GPU preferred; route=Python Core AI DiT fixed-input calls versus the independent same-weight source replay on MPS; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=77f49a8ac5bc7dd0af04766a8d3f5fbd5ede4431c6eccaa6531336eb36edf7f2; code_revision=be4cdb03c7442a5b1bc27d228e11c1190bd2facc; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal |

### 導入・制限・進捗

- [変換手順（英語）](../../Tools/Conversion/recipes/qwen_image21/README.md)

- [モバイル計画 #222](https://github.com/oka-yuji/coreai-samples/issues/222)
<!-- model-report:end -->
