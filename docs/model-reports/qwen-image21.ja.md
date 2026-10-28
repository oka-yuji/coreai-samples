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
| ios | — | 数値合格・本人画質受入済み候補はローカル梱包済み。本体でtext推論は成功するが、rawとh18p AOTのDiT試行は完成画像を得る前にOSのメモリ圧力で終了。通常Models、公開DL、端末予算の較正は未完了。 | — |

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
| 変換・参照との一致 | 合格（記載範囲） | 現行の正式FMA profileはmacOSで固定1,230比較と最終60件が全合格し、基準を変えずraw再監査も成功。旧G64/G16不合格は別記のまま保持する。iPhone native・画質の受入は示さない。 | [E13](../research/qwen-image21-g16-20261026.json) |
| 用途の品質 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 連続利用・資源解放 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 停止・再試行・回復 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](../research/qwen-image21-20261018/README.md) | 2026-10-18 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 512×512 RGBA、batch1、CFG1、8sigma。生成と参照画像1枚の編集。Python診断host、段階ロードとtensor取得を含む。 · variant=FP32; cache off; backend=Core AI; GPU preferred; route=Python conversion / diagnostic host; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; batch_size=1; concurrency=1; 未記録: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E2](../research/qwen-image21-off-on-20261021.json) | 2026-10-21 / standard / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 固定e01編集、seed20261018、512×512 RGBA、batch1、CFG1、保存8sigma。hash/prepareからPNG保存まで、段階ロード込み。診断modeごとに新process1回。cache hit自体は独立に観測していない。 · variant=FP32; no prefix KV; diagnostics off; backend=Core AI; GPU preferred; route=Python Core AI; separate process; pre-existing cache entries; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=43306970b99dda7717f7e8500179204fb0d0a9503c5836144b15b6a1d48b5852; code_revision=0a3480f617240ff43911c0a3f9a82b12cac1b8fa; batch_size=1; concurrency=1; warmups=0; power=AC; thermal=nominal |
| [E3](../research/qwen-image21-off-on-20261021.json) | 2026-10-21 / standard / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 固定e01編集、seed20261018、512×512 RGBA、batch1、CFG1、保存8sigma。hash/prepareからPNG保存まで、段階ロード込み。診断modeごとに新process1回。cache hit自体は独立に観測していない。 · variant=FP32; no prefix KV; diagnostics trace; backend=Core AI; GPU preferred; route=Python Core AI; separate process; pre-existing cache entries; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=43306970b99dda7717f7e8500179204fb0d0a9503c5836144b15b6a1d48b5852; code_revision=0a3480f617240ff43911c0a3f9a82b12cac1b8fa; batch_size=1; concurrency=1; warmups=0; power=AC; thermal=nominal |
| [E4](../research/qwen-image21-w4-smoke-20261022.json) | 2026-10-22 / legacy / diagnostic | 未記録 / 未記録 / 未記録 | 固定g01/e01、seed20261018、512正方形、8step。資産revisionは5部品記録のhashで検証済みbundleではない。電源/熱は未記録で速度の主張には使わない。 · variant=qmi-mobile-w4-v1; DiT F16 math; other components F32 math; backend=Core AI; GPU preferred; route=Python Core AI fixed-input and captured rollout versus original Diffusers on MPS; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=ce6e70483c6a47f456fb1a4bd92e95fe6993b31fada74e460311eaf90f2f77e4; code_revision=ec5f51096bb1a3f41d938045163bebdd5c76d14c; 未記録: device, os, toolchain, batch_size, concurrency, warmups, power, thermal |
| [E5](../research/qwen-image21-w4-f32-diagnostic-20261022.json) | 2026-10-22 / legacy / diagnostic | 未記録 / 未記録 / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | 凍結60run suite、8step。原典Bは全60runと必須2反復を完了し、反復差は0。Core AIの固定比較は1,230件中32件がfull valid jointで不合格（e08/e09の20261018・20261021、各8step）、target領域480件は全合格。自由rolloutは60run中16runで中止。JSONに書出し資産のinventory hashはなく、電源/熱は未記録で速度の主張には使わない。 · variant=qmi-mobile-w4-f32-v1; F32 math for all components; backend=Core AI; GPU preferred; route=Python Core AI fixed-input comparisons and partial free rollout versus the independent same-weight source reference B on MPS; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; code_revision=f5cca6f12dde207fa9a9abc6c4af6eb31195951d; 未記録: device, os, artifact_revision, batch_size, concurrency, warmups, power, thermal |
| [E6](../research/qwen-image21-selective-dit-20261022.json) | 2026-10-22 / legacy / diagnostic | M4 Max / macOS 27.0 / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | W4/F32 cohortの固定DiT step0入力4件（g01/e01/e08/e09、20261018）を再利用。0始まりのDiT block16〜23のLinear weightをF16で保持するw4-blocks16-23-f32はfull-joint/targetとも4/4合格、w4-blocks20-21-f32はfull-jointで2/4（e08/e09）不合格。資産revisionは8block DiTのmain.mlirbのSHA-256で、他の資産hashはJSONにある。60run suite・rollout・品質・実機の結果ではなくverified=false。電源/熱は未記録で速度の主張には使わない。 · variant=w4-blocks16-23-f32 and w4-blocks20-21-f32; DiT only; F32 math; listed DiT blocks keep F16 Linear weights; backend=Core AI; GPU preferred; route=Python Core AI DiT fixed-input calls versus the independent same-weight source replay on MPS; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=77f49a8ac5bc7dd0af04766a8d3f5fbd5ede4431c6eccaa6531336eb36edf7f2; code_revision=be4cdb03c7442a5b1bc27d228e11c1190bd2facc; 未記録: batch_size, concurrency, warmups, power, thermal |
| [E7](../research/qwen-image21-attention-20261026.json) | 2026-10-26 / legacy / diagnostic | M4 Max / macOS 27.0 (26A428) / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | B全60runと2反復完了。固定1,230比較中32件がe12/e13・両seed・全8stepのfull-joint nmaxで不合格。target480と他270は合格。rollout3件合格後、原因調査のため中止し57件未完。rawとbindingを独立監査。電源/熱未記録で速度の主張はしない。 · variant=w4-blocks16-23-f32, regular SDPA, all F32 math; backend=Core AI; GPU preferred; route=Saved independent same-weight B on MPS versus Core AI; private contrasts explicitly identified; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=50fba05228831e56803f3e4b80d52ff9a628a63805c5dff070e548e8365a2ed1; code_revision=23f38c7e21689918768245d8bb39f23c10e99dbb; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal |
| [E8](../research/qwen-image21-attention-20261026.json) | 2026-10-26 / legacy / diagnostic | M4 Max / macOS 27.0 (26A428) / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | 私有script hashと資産inventoryはJSONに記録。step0の6入力はTorch/Core AI双方で合格。全60step0・前回失敗32step・以前target誤差が大きかったstepを含む拡張95件も合格。正式修正後の全評価・画質・実機の合格ではない。メモリ値は別processのMac生涯peakで、phase peak/端末budgetではない。電源/熱は未記録。 · variant=private explicit F32 attention with safe256-query chunks; unchanged effective weights/full ABI; backend=Core AI; GPU preferred; route=Saved independent same-weight B on MPS versus Core AI; private contrasts explicitly identified; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=19dfb1193167a6f2c0ef4e3e2d3687e5fcefc02aa37eef7edc04b8d4db266482; code_revision=23f38c7e21689918768245d8bb39f23c10e99dbb; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal |
| [E9](../research/qwen-image21-attention-20261026.json) | 2026-10-26 / legacy / diagnostic | M4 Max / macOS 27.0 (26A428) / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | Bは60runと2反復を完了し、最終hashは旧Bと一致。固定1,230比較は全合格。SIGKILL前46runと同一code/部品の追試14runを合わせてraw指標を再計算し、自由rollout59件合格、g08-20261021不合格（latent nRMSE 0.01758714、RGBA RMSE 0.02754416）を確認。軌道外step2/3も同入力比較は合格し、その2点では原典の入力感度が支配的。各別processの4rollout対照ではautorelease poolでPythonのメモリ増加は解消しなかった。別IDのDiT group16候補の実測は別記。品質・実機・速度の受入ではない。 · variant=w4-blocks16-23-f32, safe256 explicit F32 attention; backend=Core AI; GPU preferred; route=Saved independent same-weight B on MPS versus Core AI; private contrasts explicitly identified; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=4cd8ccf4af76e85abb37de69b85c1931256fd93d929cf1923a7442dee6cc851e; code_revision=1f1d9b42625d920f7e7ec74a657f52926a939fa3; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal |
| [E10](../research/qwen-image21-g16-20261026.json) | 2026-10-26 / legacy / diagnostic | M4 Max / macOS 27.0 (26A428) / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | B全60件と2反復、Cは1caseごとの別processで全固定1,230比較・自由rolloutを完了。最終latent/RGBA全60件とtarget480比較は合格したが、e10/e11両seedのfull-joint最大誤差28件が不合格（最大nmax0.00120361、上限0.001）。source/asset inventoryとraw指標を再監査。全profile・品質・端末UIの受入ではない。 · variant=w4-dit-g16-blocks16-23-f32; safe256; unchanged frozen gate; backend=Core AI; GPU preferred; route=Saved independent same-weight B on MPS versus Core AI; private contrasts explicitly identified; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=bb7ba7f1fd1f5a0cc48c9aebda37c957f98dc1aa952263e669f2e7237cb198e6; code_revision=c2d6a5df4c69b51fc6fcfb328ab30ed38c507deb; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal |
| [E11](../research/qwen-image21-g16-20261026.json) | 2026-10-26 / legacy / diagnostic | iPhone 17 Pro (iPhone18,1) / iOS 27.0.1 (24A446) / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | 現行G16 DiTだけ。g08-20261021生成/e12-20261018編集のstep0を各2cycle。全physical出力とfull-valid/target計8比較を独立再計算し合格。ロード・実行・解放・再ロード完了。生涯peak footprint1,571,167,568 Bで、phase budget/全native経路の証拠ではない。診断copy・cache不明を含むため速度を主張しない。 · variant=w4-dit-g16-blocks16-23-f32; safe256; unchanged frozen gate; backend=Core AI; GPU preferred; route=Private signed Swift component harness; two load/call/release cycles; no normal Models workflow; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=40eb84df5f6357086e88aec1308b2eb1ac5f94a295dcae04a49cc227196d24fd; code_revision=c2d6a5df4c69b51fc6fcfb328ab30ed38c507deb; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal |
| [E12](../research/qwen-image21-g16-20261026.json) | 2026-10-26 / legacy / diagnostic | M4 Max / macOS 27.0 (26A428) / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | 新B60件と2反復が完了し、全60の最終latent/RGBA/PNGは旧G16 Bとbyte一致。5部品×2callの全18比較が合格し、Vision全4出力と旧最大誤差のe10-20261018 DiT step3（full-joint nmax0.000802730）を含む。この事前検査時点では全評価は未完で、後続の全数値合格は別記。別のiPhone試行はvm-pageshortageで終了し、終端JSON/rawなし。loader変更と端末状態の寄与は未分離で、mobile profileの受入ではない。 · variant=w4-dit-g16-blocks16-23-f32; safe256; explicit Metal FMA RoPE; backend=Core AI; GPU preferred; route=Ten fixed component calls with fresh formal FMA assets and their own matched B; host preflight for device diagnostics; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=b09a43c49f6dee9f6815d0d2adfcee2b927e5289e253a015b901a79c184208c5; code_revision=e403d2001465026f797411b1dea31355d0be4808; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal |
| [E13](../research/qwen-image21-g16-20261026.json) | 2026-10-26 / legacy / diagnostic | M4 Max / macOS 27.0 (26A428) / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | 同weight B60件＋2反復、固定1,230比較、自由60件の最終latent/RGBAが凍結基準に全合格。source/export inventoryと全raw指標を再照合し、public auditも成功。旧候補の不合格を保持する。macOSの数値受入であり、人の画質、iPhone安定性/予算、通常Models操作は未受入。 · variant=w4-dit-g16-blocks16-23-f32; safe256; explicit Metal FMA RoPE; backend=Core AI; GPU preferred; route=Complete same-weight original B versus Core AI, each C case in a fresh process; all raw metrics independently audited; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=b09a43c49f6dee9f6815d0d2adfcee2b927e5289e253a015b901a79c184208c5; code_revision=e403d2001465026f797411b1dea31355d0be4808; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal |
| [E14](../research/qwen-image21-native-complex-20261029.json) | 2026-10-29 / legacy / diagnostic | M4 Max / macOS 27.0 (26A428) / Core AI Torch 0.4.3; Core AI 1.0.0b3; Torch 2.9.0 | g08/e10/e11×2seed×8stepの固定DiT全48出力が凍結full-valid/target基準に合格し、受入済みFMA出力とbit一致。Macのfresh process/load-only footprintは3,284,228,000から772,331,304 B、Metal allocatedは3,219,095,552から705,314,816 Bへ低下。5部品の実圧縮記録は不変。全suite、画質受入の対応付け、iPhoneは未完。 · variant=qmi-mobile-w4-dit-g16-native-complex-blocks16-23-f32-v1; backend=Core AI; GPU preferred; route=Selected fixed-input Core AI preflight and separate fresh-process Swift load-only comparison; source_revision=fc13ba1986fcb905854dc8de2f52eee395f49b5b; artifact_revision=eacded0298a54afb1980c329c414e53e0b954b79ef905696712b310166eca568; code_revision=b72011063e8e1c21c1bf89a6a3dc120b6a53048b; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal |

### 導入・制限・進捗

- [変換手順（英語）](../../Tools/Conversion/recipes/qwen_image21/README.md)

- [モバイル計画 #222](https://github.com/oka-yuji/coreai-samples/issues/222)

- [本人画質受入済みローカルbundle](../research/qwen-image21-owner-bundle-20261028.json)

- [本体アプリの実機診断記録](../research/qwen-image21-native-checkpoints-20261029.json)

- [AOT資産の実機対照](../research/qwen-image21-aot-20261029.json)

- [組込み複素数候補の事前検査](../research/qwen-image21-native-complex-20261029.json)
<!-- model-report:end -->
