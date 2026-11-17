# VoxCPM2

[トップ](../../README.ja.md) · [English](voxcpm2.md)

音声生成

OpenBMB VoxCPM2 の 6 つの FP32 の Core AI 資産は、記録した基準の範囲で元の部品と一致し、文章からの FP32 の生成を Python のホストで確かめました。アプリ用の bundle を 3 つ(FP32、FP16、`proj` だけ FP32 の FP16)作りましたが、まだアプリからは読んでいません。既定の構成は聴き比べの後に決めます。

この音声は、同じ重みを Core AI で実行したものです。モデルの内部の丸めの境目の近くの数値の差で、同じ seed でも参照の実装と生成の結果が変わる場合があります。

FP16 の資産を含む構成は、FP32 と別の軌道の音声になります。停止したパッチが FP32 と一致したのは 15 回中 10 回です。音質の聴き比べは未実施です。

確かめていないこと: iPhone、Mac のアプリでの動作、参照音声(声のクローン)、逐次の再生、Neural Engine、8bit / 4bit、一般の文と seed での挙動、音質の定量評価。

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](../benchmarking.ja.md) · [元の記録](records/voxcpm2.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | — | ネイティブデモは未提供。Pythonのホストの結果はアプリの対応ではない。 | — |
| ios | — | ネイティブデモは未提供。Pythonのホストの結果はアプリの対応ではない。 | — |

### 性能と品質

| 指標 | 結果 | 集計・件数 | 条件と根拠 |
|---|---|---|---|
| バンドル・資産容量 | 9,349,325,520 B | single; n=1 | [E1](../validation/voxcpm2-conversion-20261117.md) · fp32のbundle。manifest・tokenizer・license・noticeを含む22ファイル。 |
| バンドル・資産容量 | 4,677,413,664 B | single; n=1 | [E1](../validation/voxcpm2-conversion-20261117.md) · fp16のbundle。22ファイル。FP16の結果は記録だけ。 |
| バンドル・資産容量 | 4,715,188,166 B | single; n=1 | [E1](../validation/voxcpm2-conversion-20261117.md) · fp16_proj32のbundle(projだけFP32)。22ファイル。FP16の結果は記録だけ。 |
| 特殊化キャッシュ容量 | 未計測 | — | 未計測。特殊化のcacheの大きさは記録していない。 |
| 初回ロード（特殊化を含む） | 未計測 | — | 定義どおりには未計測。メモリの計測のロード時間(fp32 7.07秒、fp16 3.40秒)は特殊化のcacheの状態を制御していない。 |
| 特殊化済みキャッシュからのロード | 未計測 | — | 定義どおりには未計測。メモリの計測のロード時間は特殊化のcacheの状態を制御していない。 |
| 要求・パイプライン完了時間 | 7.81934967 s | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp32。新しいプロセスで最初のlongの生成(音声16.8秒)。7.07秒のロードを除く。 |
| 要求・パイプライン完了時間 | 5.10555254 s | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp16。新しいプロセスで最初のlongの生成(音声16.8秒)。3.40秒のロードを除く。 |
| ピークRSS | 22,462,464,000 B | max; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp32。プロセスの生涯の最大常駐(getrusage)。ファイルにmapした重みのページを含む。Pythonのホストで、アプリのメモリではない。 |
| ピークRSS | 11,290,116,096 B | max; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp16。プロセスの生涯の最大常駐(getrusage)。Pythonのホストで、アプリのメモリではない。 |
| ピークphysical footprint | 842,347,816 B | max; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp32。カーネルが記録したプロセスの生涯の最大physical footprint。Pythonのホストで、アプリのメモリではない。 |
| ピークphysical footprint | 662,626,224 B | max; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp16。カーネルが記録したプロセスの生涯の最大physical footprint。Pythonのホストで、アプリのメモリではない。 |
| 推論中の常駐メモリ | 16,731,029,504 B | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp32。資産を保持したまま生成の後に採取した常駐サイズ。 |
| 推論中の常駐メモリ | 8,580,218,880 B | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp16。資産を保持したまま生成の後に採取した常駐サイズ。 |
| 解放後メモリの基準との差 | 670,462,648 B | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp32。部品を閉じた後のphysical footprintからロード前を引いた値。その時点でシステムのwiredは減っていない(根拠を参照)。 |
| 解放後メモリの基準との差 | 533,831,024 B | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp16。部品を閉じた後のphysical footprintからロード前を引いた値。その時点でシステムのwiredは減っていない(根拠を参照)。 |
| 課題達成率 | 未計測 | — | 未計測。用途の品質の評価セットはない。 |
| 参照との誤差（定義は根拠参照） | 4.62576051e-06 error | max; n=2151 | [E1](../validation/voxcpm2-conversion-20261117.md) · FP32の部品の最大誤差(最大絶対誤差/参照の最大絶対値)、dit.main。proj.fuseは分類で判定(一致233・境界整合1・不合格0)。nはdit.mainの呼び出し数。 |
| 停止受付から完了までの時間 | 未計測 | — | 未計測。アプリと停止の経路はまだない。 |
| 再生可能な最初の音声までの時間 | 未計測 | — | 未計測。Pythonのホストは逐次に再生せず、bundleを読むアプリもまだない。 |
| RTF（生成時間／音声長） | 0.455897562 ratio | median; n=15 | [E2](../validation/voxcpm2-conversion-20261117.md) · fp32。音声1秒あたりの生成秒数。warm-up 1回の後。ロードと保存を除く。 |
| RTF（生成時間／音声長） | 0.294079473 ratio | median; n=15 | [E2](../validation/voxcpm2-conversion-20261117.md) · fp16。同上。FP32と別の軌道の音声。試聴は未完了。 |
| RTF（生成時間／音声長） | 0.294309107 ratio | median; n=15 | [E2](../validation/voxcpm2-conversion-20261117.md) · fp16_proj32。同上。FP32と別の軌道の音声。試聴は未完了。 |
| 文字誤り率CER | 未計測 | — | 未計測。音声認識による評価はしていない。 |
| 人による品質評点（採点基準を固定） | 未計測 | — | 未計測。聴き比べは未実施。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 合格（記載範囲） | FP32の部品、置き換え3パッチでの5文のFP32の生成、FP32のbundleの再照合が合格。FP16は記録だけ。試聴は未完了。 | [E1](../validation/voxcpm2-conversion-20261117.md) |
| 用途の品質 | 未計測 | 品質の評価は未実施。既定の構成は聴き比べの後に決める。 | — |
| 連続利用・資源解放 | 未計測 | アプリでの連続利用と解放の確認は未実施。 | — |
| 停止・再試行・回復 | 未計測 | 停止と再試行の確認は未実施。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](../validation/voxcpm2-conversion-20261117.md) | 2026-11-17 / legacy / conversion | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Python 3.11.9; torch 2.9.0; transformers 4.57.3; coreai-core 1.0.0b3; coreai-torch 0.4.3 | 固定の5文・seed 1708。合否に使う部品の照合3,934件、文章からのFP32の生成(置き換えは各文1パッチ以下、239中3)、デバッグ情報を除いたFP32のbundleでの同じ照合。アプリではなくPythonのホストだけ。 · variant=fp32 bundle; backend=Python Core AI host, GPU-preferred specialization; compute units not profiled; route=VoxCPM2 recipe: original FP32 CPU reference -> component comparisons -> FP32 generation from text -> debug-information removal -> FP32 bundle re-check; source_revision=7936080b41163874f38610f46f438e2c0757f9fd; artifact_revision=f771b72d2b18eb01b0a832cadee4bfff37ca7c76a3627defbf1f868d158d70e8; batch_size=1; concurrency=1; 未記録: code_revision, warmups, power, thermal, plan_sha256 |
| [E2](../validation/voxcpm2-conversion-20261117.md) | 2026-11-17 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Python 3.11.9; torch 2.9.0; transformers 4.57.3; coreai-core 1.0.0b3; coreai-torch 0.4.3 | 構成ごとに5文×seed 1708〜1710。各生成は同じworkerでのwarm-up 1回の後に計時。load average 4.1〜6.1。合否なしの記録。bundleはデバッグ情報を除いた同じprogram。 · variant=fp32, fp16 and fp16_proj32 configurations; backend=Python Core AI host, GPU-preferred specialization; compute units not profiled; route=VoxCPM2 recipe synthesis_compare: exported assets, one worker process per generation; source_revision=7936080b41163874f38610f46f438e2c0757f9fd; batch_size=1; concurrency=1; warmups=1; 未記録: artifact_revision, code_revision, power, thermal, plan_sha256 |
| [E3](../validation/voxcpm2-conversion-20261117.md) | 2026-11-17 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Python 3.11.9; torch 2.9.0; transformers 4.57.3; coreai-core 1.0.0b3; coreai-torch 0.4.3 | 文long(97 token、105パッチ、音声16.8秒)、seed 1708、プロセスで最初の生成。ロードの前後、生成の後、解放の後にプロセスのメモリを、リポジトリの外の1回限りのスクリプトで採取。合否なしの記録で、アプリのメモリではない。 · variant=fp32 and fp16 bundles; backend=Python Core AI host, GPU-preferred specialization; compute units not profiled; route=One new Python process per bundle: load -> one generation of long -> release, under the shared heavy-job lock; source_revision=7936080b41163874f38610f46f438e2c0757f9fd; artifact_revision=bundles of 20261116T204724-264d1994 (same main.mlirb and main.hash as the current fp32 and fp16 bundles); batch_size=1; concurrency=1; warmups=0; 未記録: code_revision, power, thermal, plan_sha256 |

### 導入・制限・進捗

- [#573](https://github.com/oka-yuji/coreai-samples/issues/573)

- [変換手順](../../Tools/Conversion/recipes/voxcpm2/README.md)

- [変換とbundleの検証記録](../validation/voxcpm2-conversion-20261117.md)
<!-- model-report:end -->
