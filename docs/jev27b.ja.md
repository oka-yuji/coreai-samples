# JEV-27B の型付き判断(Core AI 版)

[トップ](../README.ja.md) · [English](jev27b.md) · [Hugging Face から取得](https://huggingface.co/okayuji/jev-27b-coreai)

JEV-27B は、状況についての型付きの質問に答えるモデルです。質問の型は `noul`(false か true)、`choice`(2〜16 択)、`score`(0〜5)の 3 つです。このページは、[`jev27b` の変換レシピ](../Tools/Conversion/recipes/jev27b/README.md)が [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) から作る Core AI の bundle と、その記録した検査を説明します。素のテンプレートを 1 回 prefill して Core AI の資産で最後のトークンの隠れ状態を出し、24 行の判断ヘッドで選択肢の確率に変えます。bundle は Hugging Face から取得し、Mac の `coreai-verify typed-decide` で動かせます。サンプルアプリへの組み込みは別の作業です。

**向いている人:** Swift から Mac 上の型付き判断を動かし、測りたい開発者。**向かない用途:** 自由な対話、本番の判断、追加の検査なしに確率を現実の正しさとして使うこと。

必要なもの: Apple silicon、macOS 27、Metal Toolchain を入れた Xcode 27、GPU での実行。計測は Mac16,6(Apple M4 Max、メモリ 128 GiB)、macOS 27.0(26A428)で行いました。最小のメモリ構成は確かめていません。配布済み bundle の利用には変換は不要です。自分で bundle を作るには、レシピの参照環境と変換環境、固定した入力、空きディスクが要ります(書き出しの段は 160 GB の空きを確かめます)。

## 取得して動かす

Rust/Cargo を含む[共通の準備](../README.ja.md#はじめる)と Hugging Face CLI の導入後、リポジトリのルートで実行します。

~~~sh
hf download okayuji/jev-27b-coreai --local-dir models/jev27b
./Tools/Irodori/build-inputs.sh
swift build -c release --package-path Tools/Verify --product coreai-verify --force-resolved-versions
Tools/Verify/.build/release/coreai-verify typed-decide --bundle models/jev27b/bundle --input models/jev27b/examples/request.json --output jev-decision-output.json
~~~

出力ファイルには、まだ存在しない名前を使います。モデルを動かすときは `AGENTS.md` の共有ロックを取ります。
取得先の `bundle/` を指定してください。HF の取得キャッシュ・カード・例は、その外側に置く構成です。
公開用に書いた入力例と実行結果はモデルカードに、評価済み版と公開版の対応は[公開記録](validation/jev27b/release.md)にあります。

## 実行部のメモリ安定化（2026-11-05）

同じ形状の間は実行部を保持し、4回目の形状変更前にmodelとfunctionを解放して読み直します。再loadまたはbackbone推論中の失敗・取消では、実行部を破棄します。公開済みの重み・head・tokenizer・温度は変更していません。長さ交替100判断・同長100判断・近い長さの交替20判断は、全出力を保って固定メモリ条件に合格しました。有限の検査であり、無制限利用の保証ではありません。再loadと初回実行の時間が加わります。下の既存の時間は対策前の測定を保持したもので、改善前後・元モデルの同条件比較は別に行います。[測定条件と結果](validation/jev27b/README.md#memory-stability-2026-11-05)。

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](benchmarking.ja.md) · [元の記録](model-reports/records/jev27b.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | — | bundle は coreai-verify のコマンドラインでだけ動く。アプリへの組み込みは #449。 | — |
| ios | — | このOSのネイティブデモは未提供です。 | — |

### 性能と品質

| 指標 | 結果 | 集計・件数 | 条件と根拠 |
|---|---|---|---|
| バンドル・資産容量 | 16,064,815,496 B | single; n=1 | [E5](validation/jev27b/README.md#bundle) · 校正済み bundle の 13 ファイル(16 GB の資産を含む)。計画の改訂 3 の後に performance.py records で数えた。 |
| 特殊化キャッシュ容量 | 未計測 | — | 特殊化 cache の大きさはコマンドラインのツールから見えないので測っていない。 |
| 初回ロード（特殊化を含む） | 28.2243533 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · 特殊化 cache を消した新プロセス(--clear-cache)。bundle の検査・tokenizer・ヘッド・本体の合計。ファイルの cache は消していない。AC 給電(140 W アダプター、94 W)、エネルギーモード High Power(plan-d2b-ac94)。 |
| 特殊化済みキャッシュからのロード | 9.50495433 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · 特殊化 cache がある新プロセス。初回と同じ部分の合計。AC 給電(140 W アダプター、94 W)、エネルギーモード High Power(plan-d2b-ac94)。 |
| 要求・パイプライン完了時間 | 0.614704083 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · ちょうど 100 トークンの P の行。TypedDecisionModel.decide の呼び出しから復帰まで。1 プロセスで長さを交互に、各長さ warmup 1 回の後。各行の run 中に 100 ms ごとにメモリを採取(--sample-memory)。各行は熱の状態 nominal で開始(--wait-nominal-thermal 600)。第 3 周の行は熱の状態 nominal を 54.51 s 待ってから始めた。AC 給電(140 W アダプター、94 W)、エネルギーモード High Power(plan-d2b-ac94)。 |
| 要求・パイプライン完了時間 | 2.20761833 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · ちょうど 500 トークンの P の行。TypedDecisionModel.decide の呼び出しから復帰まで。1 プロセスで長さを交互に、各長さ warmup 1 回の後。各行の run 中に 100 ms ごとにメモリを採取(--sample-memory)。各行は熱の状態 nominal で開始(--wait-nominal-thermal 600)。AC 給電(140 W アダプター、94 W)、エネルギーモード High Power(plan-d2b-ac94)。 |
| 要求・パイプライン完了時間 | 8.97832325 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · ちょうど 2,000 トークンの P の行。TypedDecisionModel.decide の呼び出しから復帰まで。1 プロセスで長さを交互に、各長さ warmup 1 回の後。各行の run 中に 100 ms ごとにメモリを採取(--sample-memory)。各行は熱の状態 nominal で開始(--wait-nominal-thermal 600)。AC 給電(140 W アダプター、94 W)、エネルギーモード High Power(plan-d2b-ac94)。 |
| 要求・パイプライン完了時間 | 37.1391402 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · ちょうど 8,192 トークンの P の行。TypedDecisionModel.decide の呼び出しから復帰まで。1 プロセスで長さを交互に、各長さ warmup 1 回の後。各行の run 中に 100 ms ごとにメモリを採取(--sample-memory)。各行は熱の状態 nominal で開始(--wait-nominal-thermal 600)。第 2 周の行は始めて約 20 s 後に熱の圧力 heavy になった。中央値の行(第 1 周)は nominal の下。AC 給電(140 W アダプター、94 W)、エネルギーモード High Power(plan-d2b-ac94)。 |
| ピークRSS | 32,333,840,384 B | single; n=1 | [E4](validation/jev27b/README.md#ac-94-w) · latency のプロセスの /usr/bin/time -l。読み込みを含む生涯の最大。footprint と足さない。AC 給電(140 W アダプター、94 W)、エネルギーモード High Power(plan-d2b-ac94)。 |
| ピークRSS | 32,646,430,720 B | single; n=1 | [E8](validation/jev27b/README.md#memory-stability-2026-11-05) · warmup後の4長さの交替100判断。wait4 maximum RSS including load。速度改善の測定ではない。RSSとfootprintは合算しない。 |
| ピークRSS | 32,336,232,448 B | single; n=1 | [E9](validation/jev27b/README.md#memory-stability-2026-11-05) · warmup後の同じ8192-token入力100判断。wait4 maximum RSS including load。速度改善の測定ではない。RSSとfootprintは合算しない。 |
| ピークRSS | 32,669,925,376 B | single; n=1 | [E10](validation/jev27b/README.md#memory-stability-2026-11-05) · warmup後の6144/8192-tokenの交替20判断。wait4 maximum RSS including load。速度改善の測定ではない。RSSとfootprintは合算しない。 |
| ピークphysical footprint | 21,336,815,344 B | single; n=1 | [E4](validation/jev27b/README.md#ac-94-w) · latency のプロセスの /usr/bin/time -l(peak memory footprint)。上限ではない。長さを交互に流す 1 周ごとに 4.1〜4.5 GB 増え、頭打ちになっていない。AC 給電(140 W アダプター、94 W)、エネルギーモード High Power(plan-d2b-ac94)。 |
| ピークphysical footprint | 8,393,766,472 B | single; n=1 | [E8](validation/jev27b/README.md#memory-stability-2026-11-05) · warmup後の4長さの交替100判断。kernel lifetime footprint peak。速度改善の測定ではない。RSSとfootprintは合算しない。 |
| ピークphysical footprint | 6,827,587,384 B | single; n=1 | [E9](validation/jev27b/README.md#memory-stability-2026-11-05) · warmup後の同じ8192-token入力100判断。kernel lifetime footprint peak。速度改善の測定ではない。RSSとfootprintは合算しない。 |
| ピークphysical footprint | 9,169,450,928 B | single; n=1 | [E10](validation/jev27b/README.md#memory-stability-2026-11-05) · warmup後の6144/8192-tokenの交替20判断。kernel lifetime footprint peak。速度改善の測定ではない。RSSとfootprintは合算しない。 |
| 推論中の常駐メモリ | 17,298,096,128 B | max; n=1 | [E4](validation/jev27b/README.md#ac-94-w) · 計測した 8,192 トークンの行の run 中に 100 ms ごとに採った RSS の最大(--sample-memory)。AC 給電(140 W アダプター、94 W)、エネルギーモード High Power(plan-d2b-ac94)。 |
| 解放後メモリの基準との差 | 未計測 | — | コマンドラインのプロセスは終わるまでモデルを手放さないので、解放後の基準が無い。 |
| 課題達成率 | 0.866666667 ratio | mean; n=300 | [E2](validation/jev27b/README.md#quality) · 段 3 の Q の choice の行で top が教師の top と一致した割合(教師分布への一致で、現実の正答率ではない)。段 1′ は 0.8733。 |
| 参照との誤差（定義は根拠参照） | 0.00654150054 error | max; n=301 | [E1](validation/jev27b/README.md#fidelity) · 温度 1 の F で、Core AI と同じ量子化後の重みの PyTorch float32 の確率差の最大。 |
| 停止受付から完了までの時間 | 8.84007833 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · 試験用 backbone が forward の呼び出しを報告した後に取り消した 2,000 トークンの依頼(返金のカードの例の state を 164 回繰り返したもの。P の行ではない)。backbone は取消を伝えないので、依頼は run が返った時点で終わる。時間(8.72〜9.00 s)は取り消さない同じ依頼(8.20 s)より長く、理由は調べていない。試験 target の debug build(swift test)。熱の圧力は nominal(入力の無い 120 s を 30 s 待ってから開始)。AC 給電(140 W アダプター、94 W)、エネルギーモード High Power(plan-d2b-ac94)。 |
| 文書再利用時の完了時間 | 未計測 | — | 質問をまたいで文書の状態を再利用する経路は型付き判断にまだ無い。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 合格（記載範囲） | 温度 1 の F のすべての帯で最大 0.03・平均 0.01 の内側(最大 0.0065)。top の変化なし。 | [E1](validation/jev27b/README.md#fidelity) |
| 用途の品質 | 合格（記載範囲） | choice の正答率は点推定 −0.67 pt で閾値 −2 pt を満たすが、95% 区間 [−3.00, +1.67] は閾値をまたぐ(n = 300)。noul の Brier +0.00024、score の正規化 MAE +0.00051。教師分布への一致で、現実の正答率ではない。 | [E2](validation/jev27b/README.md#quality) |
| 連続利用・資源解放 | 合格（記載範囲） | 長さ交替100判断・同長100判断・近い長さの交替20判断で、出力を保ち固定メモリ条件に合格。4回目の形状変更前にmodel/functionを回収する。有限の検査で無制限利用の保証ではなく、再loadの時間が加わる。 | [E8](validation/jev27b/README.md#memory-stability-2026-11-05) · [E9](validation/jev27b/README.md#memory-stability-2026-11-05) · [E10](validation/jev27b/README.md#memory-stability-2026-11-05) · [E11](validation/jev27b/README.md#memory-stability-2026-11-05) |
| 停止・再試行・回復 | 合格（記載範囲） | forward の run 中に取り消した 3 回の依頼は run が返った時点で CancellationError で終わり、checkpoint での取消は run の前に終わった。どちらの取消の後も、次の行の logit は取消の前の同じ行とビット単位で同じだった。試験用 backbone は取消を伝えないので、Core AI の run 自体が止まるかは測っていない。 新Session経路もRelease診断で取消・再load・次の答えの一致を確認した。 | [E4](validation/jev27b/README.md#ac-94-w) · [E11](validation/jev27b/README.md#memory-stability-2026-11-05) |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](validation/jev27b/README.md#fidelity) | 2026-11-03 / legacy / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | F の 302 行(301 行を流し、8,192 トークン超の 1 行は拒否)を温度 1 で、同じ量子化後の重みの PyTorch float32 と入力の帯ごとに比較。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide (recipe step typed_decide_f) and the PyTorch reference stage 2; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=26a11dca07a265feca4970dccdf2ebabaa005730132035e9df37b207cd26051e; code_revision=c626c9a27f448961cc9d84addba31ce74573e183; batch_size=1; concurrency=1; 未記録: warmups, power, thermal, plan_sha256 |
| [E2](validation/jev27b/README.md#quality) | 2026-11-04 / legacy / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | Q(900 行)・Cal(3,000)・H(1,500)を教師分布と比較。量子化と適応の条件は計測前に固定。温度は Cal で再フィットし H′ で採否。教師への一致で、現実の正しさではない。結果は 2026-11-03(段 0q と 2)から 2026-11-04(ほかの段、採点、校正)。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide for stage 3 (recipe steps typed_decide_q / cal / h) and the PyTorch reference stages 0, 0q, 1' and 2; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=26a11dca07a265feca4970dccdf2ebabaa005730132035e9df37b207cd26051e; code_revision=31071af7acef18f77362609bb548233b61ad624a; batch_size=1; concurrency=1; 未記録: warmups, power, thermal, plan_sha256 |
| [E3](validation/jev27b/README.md#japanese) | 2026-11-04 / legacy / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | JGLUE の J(各 task 200 行、prompt の 2 変種、段 1′ と 3)。測定は完了。合格線は無く、passed は完了を表し基準を満たしたことではない。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide and PyTorch MPS bf16 (recipe steps typed_decide_j / reference_j); source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3; code_revision=b64eae58e9b4a8954b6b9270a109e6edd6bd22c8; batch_size=1; concurrency=1; 未記録: warmups, power, thermal, plan_sha256 |
| [E4](validation/jev27b/README.md#ac-94-w) | 2026-11-04 / standard / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | P はちょうど 100・500・2,000・8,192 トークンの 4 行。1 判断の時間は 1 プロセスで長さごとに warmup 1 回と計測 3 周で、各行を熱の状態 nominal で始めた。ロードは新しいプロセスで fresh 3 回と cached 3 回。メモリは /usr/bin/time -l と 100 ms の採取。取消は 2,000 トークンの依頼で 3 回。AC 給電(140W USB-C 電源アダプター、94 W。残量 100%、充電済み)、本人の設定のエネルギーモード High Power で計測した(plan-d2b-ac94)。各段は 94 W、熱の圧力 level 0、入力の無い 120 s 以上、音声・動画・自動操作の assertions なしで始めた。入力は 1 判断の時間のプロセスの間にあり、最後は 11:26:13 ごろで熱の状態 nominal を待つ間(行は流れていない)。それより前の入力は、HIDIdleTime が最後の 1 回しか残さないので除けない。第 2 周の 8,192 トークンの行は始めて約 20 s 後に heavy になった。warmup は 1 判断の時間の周だけで、ロードと取消には無い。他の負荷: モデルの他のプロセスなし(計測の前の pgrep)。ほかのプロセスの CPU と GPU の使用は記録していない。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide --wait-nominal-thermal 600 (release build); TypedDecisionInferenceTests in the debug build of the test target for cancellation; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3; code_revision=18890f539107dd14a65458981db7853d2d959dcb; batch_size=1; concurrency=1; warmups=1; power=AC: 140W USB-C Power Adapter, 94 W (pmset -g ac); battery 100% charged; energy mode High Power (powermode 2 in pmset -g); thermal=pmset -g therm: no warning; ProcessInfo thermalState nominal at the start of every row (the longest wait 54.51 s) and thermalmonitord pressure (mTPL) 0 at the start of every stage; during the latency process moderate from 11:25:51 and heavy 11:25:53-11:26:52 (the 8,192-token row of the second round) |
| [E5](validation/jev27b/README.md#bundle) | 2026-11-04 / standard / conversion | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | 校正済み bundle のファイルの大きさ。計画の改訂 3 を凍結した後に数えた。decision.json の SHA-256 は計画の artifact revision と同じ。モデルは読まない。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=performance.py records over the calibrated bundle written by the recipe step calibrate; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3; code_revision=e879c67ff30344fc69f5cb6dcde93ed432d9339c; batch_size=1; concurrency=1; warmups=0; power=not applicable (file sizes); thermal=not applicable (file sizes) |
| [E6](validation/jev27b/README.md#battery-power) | 2026-11-04 / standard / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | P はちょうど 100・500・2,000・8,192 トークンの 4 行。1 判断の時間は 1 プロセスで長さごとに warmup 1 回と計測 3 周。ロードは新しいプロセスで fresh 3 回と cached 3 回。メモリは /usr/bin/time -l と 100 ms の採取。取消は 2,000 トークンの依頼で 3 回。本人の決定によりバッテリー駆動で計測した(90% から 76%、アダプターなし、低電力モード off。計画の改訂 3 に記録)。第 3 周と取消の試験は熱の圧力 moderate から heavy の下。warmup は 1 判断の時間の周だけで、ロードと取消には無い。他の負荷: モデルの他のプロセスなし(計測の前の pgrep)。ほかのプロセスの GPU の使用は記録していない。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide (release build); TypedDecisionInferenceTests in the debug build of the test target for cancellation; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3; code_revision=9d8fdf821470168c3d92276808cb9ff1f815d7de; batch_size=1; concurrency=1; warmups=1; power=Battery power, no AC adapter attached (pmset -g ac); 90% at start, Low Power Mode off (powermode 0 in pmset -g). Measured on battery by the owner's instruction on 2026-11-04 06:20 JST; thermal=pmset -g therm: no warning in 10 snapshots; thermalmonitord pressure moderate from 06:28:52 (end of the second round) and heavy 06:29:08-06:32:08 (from the end of the third 2,000-token row through the cancellation test) |
| [E7](validation/jev27b/README.md#ac-70-w) | 2026-11-04 / standard / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | P はちょうど 100・500・2,000・8,192 トークンの 4 行。1 判断の時間は 1 プロセスで長さごとに warmup 1 回と計測 3 周。ロードは新しいプロセスで fresh 3 回と cached 3 回。メモリは /usr/bin/time -l と 100 ms の採取。取消は 2,000 トークンの依頼で 3 回。AC 給電(70W USB-C 電源アダプター、68 W。残量 64% から 67%、充電中)、本人の設定のエネルギーモード High Power で計測した(plan-d2b-ac)。各段は熱の圧力 level 0 で始めたが、1 判断の時間のプロセスの途中で heavy になった(第 2 周の 8,192 トークンの行と第 3 周)。warmup は 1 判断の時間の周だけで、ロードと取消には無い。他の負荷: モデルの他のプロセスなし(計測の前の pgrep)。pmset -g log には、バッテリー駆動の計測の間には無かったほかの活動(音声のセッション、ブラウザの動画の wake lock、UI の自動操作のセッション)が残っている。ほかのプロセスの CPU と GPU の使用は記録していない。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide (release build); TypedDecisionInferenceTests in the debug build of the test target for cancellation; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3; code_revision=9b610f88ba0b7810555384200686472e02cd11f2; batch_size=1; concurrency=1; warmups=1; power=AC: 70W USB-C Power Adapter, 68 W (pmset -g ac); battery 64% to 67%, charging; energy mode High Power (powermode 2 in pmset -g); thermal=pmset -g therm: no warning; thermalmonitord pressure (mTPL) 0 at the start of every stage (the cancellation test waited 50 s); during the latency process moderate from 09:26:46 and heavy 09:26:48-09:29:05 (the 8,192-token row of the second round and the third round) |
| [E8](validation/jev27b/README.md#memory-stability-2026-11-05) | 2026-11-05 / standard / diagnostic | MacBook Pro (14-inch, 2024), Apple M4 Max, 128 GiB unified memory (Mac16,6) / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | warmup 4判断後、4長さの交替を100判断。全logitsと両温度の確率が元とbit一致。CPU build/試験を許したメモリ安定性検査で、他モデルの同時実行はない。RSS・現在footprint・sampled peak・kernel lifetime peakを区別する。最初/最後5件の中央値と固定条件は元の記録を参照。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide --sample-memory --wait-nominal-thermal 600; normal Release; external proc_pid_rusage observer; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d; code_revision=ff2a2579e30a95a669b492f6f1eb6fca771211be; batch_size=1; concurrency=1; warmups=4; power=AC, pmset reports 68 W; battery and charging state recorded per run; thermal=ProcessInfo thermalState nominal at every row start; not a constant-temperature guarantee |
| [E9](validation/jev27b/README.md#memory-stability-2026-11-05) | 2026-11-05 / standard / diagnostic | MacBook Pro (14-inch, 2024), Apple M4 Max, 128 GiB unified memory (Mac16,6) / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | warmup 1判断後、同じ8192-token入力を100判断。全logitsと両温度の確率が元とbit一致。CPU build/試験を許したメモリ安定性検査で、他モデルの同時実行はない。RSS・現在footprint・sampled peak・kernel lifetime peakを区別する。最初/最後5件の中央値と固定条件は元の記録を参照。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide --sample-memory --wait-nominal-thermal 600; normal Release; external proc_pid_rusage observer; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d; code_revision=ff2a2579e30a95a669b492f6f1eb6fca771211be; batch_size=1; concurrency=1; warmups=1; power=AC, pmset reports 68 W; battery and charging state recorded per run; thermal=ProcessInfo thermalState nominal at every row start; not a constant-temperature guarantee |
| [E10](validation/jev27b/README.md#memory-stability-2026-11-05) | 2026-11-05 / standard / diagnostic | MacBook Pro (14-inch, 2024), Apple M4 Max, 128 GiB unified memory (Mac16,6) / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | warmup 2判断後、6144/8192-tokenの交替を20判断。全logitsと両温度の確率が元とbit一致。CPU build/試験を許したメモリ安定性検査で、他モデルの同時実行はない。RSS・現在footprint・sampled peak・kernel lifetime peakを区別する。最初/最後5件の中央値と固定条件は元の記録を参照。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide --sample-memory --wait-nominal-thermal 600; normal Release; external proc_pid_rusage observer; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d; code_revision=ff2a2579e30a95a669b492f6f1eb6fca771211be; batch_size=1; concurrency=1; warmups=2; power=AC, pmset reports 68 W; battery and charging state recorded per run; thermal=ProcessInfo thermalState nominal at every row start; not a constant-temperature guarantee |
| [E11](validation/jev27b/README.md#memory-stability-2026-11-05) | 2026-11-05 / legacy / diagnostic | MacBook Pro (14-inch, 2024), Apple M4 Max, 128 GiB unified memory (Mac16,6) / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | 診断用GPU6試験が合格。取消を無視する1472-tokenのrun終了後、Sessionが1回再loadして次の答えがbit一致した。再load失敗・busy・改変bundle拒否は別のCPU試験で確認。このsuite単独のbenchmark計画はない。 · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=TypedDecisionInferenceTests; Release with -enable-testing; six serialized GPU tests; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d; code_revision=42a9d95; batch_size=1; concurrency=1; power=AC, pmset reports 68 W; thermal=Not recorded per test; not a latency comparison; 未記録: warmups, plan_sha256 |

### 導入・制限・進捗

- [変換レシピ](../Tools/Conversion/recipes/jev27b/README.md)

- [検証の記録](validation/jev27b/README.md)

- [日本語(JGLUE)、合格線なし](jev27b.md#japanese-jglue)
<!-- model-report:end -->

## クイックスタート

1. レシピで bundle を作ります。固定した入力を `.local/conversion` に置き、setup を流し、`coreai-verify` を build して写し、レシピを流します([レシピの README](../Tools/Conversion/recipes/jev27b/README.md))。`calibrate` の段が、校正済みの bundle `jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic` をその出力のフォルダに書きます。写した bundle は、そのフォルダの中で `shasum -a 256 -c SHA256SUMS`、リポジトリの直下で `python3 -m Tools.Conversion.adapters.jev27b.decision <bundle>/decision.json --check-files` で確かめられます。
2. コマンドラインのツールで型付きの質問に答えます。モデルを動かすのは重いジョブなので、共有の lock を取ってから流します(`CLAUDE.md`)。

   ~~~sh
   swift build -c release --package-path Tools/Verify --product coreai-verify
   Tools/Verify/.build/release/coreai-verify typed-decide --bundle <bundle> --input in.json --output out.json
   ~~~

   `in.json` には答える行を書きます。たとえばモデルカードの返金の例です。

   ~~~json
   {"rows": [{"id": "refund", "kind": "noul",
              "state": "Customer says the parcel arrived damaged and wants their money back.",
              "question": "Is the customer asking for a refund?", "options": ["false", "true"]}]}
   ~~~

   `out.json` には、行ごとにトークン数、24 個のヘッドの logit、bundle の温度と温度 1 の確率、top の選択肢、判断にかかった時間が入ります([Running on Mac](jev27b.md#running-on-mac))。

## 記録した結果

数値は、レシピの run と性能の計測の記録([docs/validation/jev27b](validation/jev27b/README.md))から写しています。品質と校正は、corpus の教師分布(Jev 1.13 の出力分布)への一致を測ったもので、現実の正しさではありません。

- **一致**(固定集合 F の 302 行。301 行を流し、8,192 トークンを超える 1 行は拒否): 温度 1 で、同じ量子化後の重みの PyTorch float32 の実行との確率差は最大 0.0065、平均 0.0011 でした。8,192 トークンまでのどの入力の帯でも 0.03 と 0.01 の内側で、top の選択肢の変化はありません。詳細: [`fidelity.json`](validation/jev27b/fidelity.json)(帯ごと)と [Running on Mac](jev27b.md#running-on-mac)。
- **量子化と適応**(Q の 900 行): 温度 1 の段 3 と段 1′([段の定義](jev27b.md#reference-stages))の比較は、測る前に固定した 3 条件を満たしました。choice の正答率は −0.67 pt(閾値 −2)、noul の Brier は +0.00024(閾値 +0.01)、score の MAE / 5 は +0.00051(閾値 +0.02)です。choice の正答率は点推定で合格ですが、95% 区間 [−3.00, +1.67] は閾値をまたぎます(n = 300、差は 2 行)。適応の効果は 4 指標とも、bf16 の効果の 80% 以上が 4bit でも残りました。段 3 は KL 0.0202、top-1 の一致 0.903、choice の正答率 86.67%(段 1′ は 87.33%)、noul の AUROC 0.993 です。詳細: [Quality and calibration](jev27b.md#quality-and-calibration)。
- **校正**: Cal の段 3 の logit から再フィットした温度を 3 種類とも採用しました(noul 1.0056266、choice 1.0053791、score 1.010558)。取り分けておいた H′ で、NLL と Brier がカードの温度より悪くならなかったためです。この温度でも F の一致は合格のままです。
- **日本語**(JGLUE の J、各 task 200 行、合格線なし): 段 3 で採用した prompt の変種の値は、JNLI choice の正答率 0.855(`en`)、JNLI noul の正答率 0.920 と AUROC 0.978(`en`)、JCommonsenseQA の正答率 0.960(`ja`)、JSTS の MAE 0.529・Pearson 0.859・Spearman 0.792(`en`)です。詳細: [Japanese (JGLUE)](jev27b.md#japanese-jglue)。
- **性能**(計測 3 回の中央値。1 判断の時間は長さごとに warmup 1 回の後。140 W のアダプターで 94 W の AC 給電、エネルギーモード High Power。1 判断の時間の各行は熱の状態 nominal で開始): 1 回の判断は 100 トークンで 0.61 s、500 で 2.21 s、2,000 で 8.98 s、8,192 で 37.14 s です。読み込みは、特殊化を新しく行うと 28.22 s、cache からなら 9.50 s です。1 判断の時間を測ったプロセスのピークは RSS 32.3 GB、physical footprint 21.3 GB(頭打ちになっていない)で、8,192 トークンの行の間は約 17.3 GB が常駐しました。試験用の backbone で 2,000 トークンの run の最中に取り消した依頼は、取消の要求から 8.84 s 後、run が返った時点で終わりました。中央値の行は熱の圧力 nominal の下でした。先に行ったバッテリー駆動と 70 W のアダプターでの計測も、条件つきで残しています。詳細: [Performance](jev27b.md#performance)。

## 検証した範囲と上限

- 統合した checkpoint は出力の検査を満たしていません。段 1(統合前のアダプター)と段 1′(統合した checkpoint)の差は最大 0.038 で、閾値 0.01 を超えます。本人がこれを記録して統合した checkpoint で進めると決め、限られた探索として扱っています。閾値は変えていません。
- bundle は gated delta update を float32 で計算する変種(`export_gdnfp32`)で、量子化後の重みは最初の bundle と同じです。最初の bundle は 512 トークンを超える帯で一致の基準を超えました(4,097〜8,192 の帯で最大 0.138、#475)。変種はすべての帯で基準の内側です。
- 品質の条件は、蒸留の corpus から引いた固定集合で、段 3 と段 1′、適応した段と基底の段を比べたものです。教師への一致であり、人のラベルに対する正答率ではありません。choice の正答率の条件は点推定(−0.67 pt、閾値 −2)で合格ですが、95% 区間 [−3.00, +1.67] は閾値をまたぎます(n = 300、差は 2 行)。適応の効果の条件(4bit の効果が bf16 の効果の 0.8 倍以上)は 4 指標とも満たしました(割合 0.96〜1.12)。
- 採用した温度は、Cal で再フィットし H′ で教師分布に照らして選んだものです。出荷時の確率は教師に合わせた校正で、現実の正しさに対する校正ではありません。
- 日本語の結果に合格線はなく、日本語の品質や校正の合格を示すものではありません。prompt の変種は、J を流す前に別の集合で採用を決めました。
- 入力は 8,192 トークン(描画したテンプレート全体)まで受け付け、超える行は forward なしで拒否します。すべての帯を含むのは一致の集合 F だけです(512 まで 269 行、513〜2,048 が 11 行、2,049〜4,096 が 10 行、4,097〜8,192 が 11 行)。品質・校正・日本語の集合は短く、最長は Q が 593 トークン(900 行のうち 897 行が 512 以下)、Cal が 672、H が 689、J が wrapper を当てて 129 です。約 700 トークンを超える入力の品質と校正は測っていません。性能用の行はちょうど 100、500、2,000、8,192 トークンです。
- physical footprint のピーク 21.3 GB は、1 プロセスで長さを交互に 16 回流した後の値で、上限ではありません。8,192 トークンの行で 8.2 から 12.5、16.6、21.2 GB と 1 周ごとに 4.1〜4.5 GB 増え、頭打ちになっていません。それらの行の RSS は約 17.3 GB のままでした。先に行った 2 つの計測でも同じように増えました。
- forward の run の最中に取り消した依頼は、run が返るまでモデルを使用中のままにし、その後 `CancellationError` で終わります。run の前に取り消した依頼は、すぐに終わります。取消の試験は取消を伝えない試験用の backbone で forward を流すので、Core AI の run 自体が取消で止まるかは測っていません。
- 確かめていないもの: iOS、アプリへの組み込み(#449)、メモリの少ない Mac、同じプロセスでの解放と再読み込み、質問をまたいだ文書の再利用。

## 入力とファイル

- 入力は `state`、`question`、`kind`(`noul` / `choice` / `score`)、`options` です。`choice` は 2〜16 個の重ならない 1 行の選択肢、`noul` は `false` と `true`、`score` は `0`〜`5` だけを受け付けます。描画は JEV-27B の `bare-v1` テンプレートに従い、規則に合わない入力は描き換えずに拒否します。
- bundle は、`main` だけを持つ Core AI の資産(最後の RMSNorm の後の隠れ状態を返す)、`head.safetensors`(`proj.weight` [24, 5120] と `proj.bias` [24]、float32)、配布元の tokenizer のバイト単位の写し、`decision.json`(`typed-decision-v1`)、書き出しが書く `metadata.json`、`LICENSE`、`NOTICE.md`、`SHA256SUMS` からなります。校正済みの bundle には、温度の再フィットの記録 `calibration-refit.json` が加わります。
- 評価時の校正済み bundle（公開用の表示変更前）は 13 ファイル、16,064,815,496 バイトで、`decision.json` の SHA-256 は `a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3` でした。配布する `bundle/` は 13 ファイル、16,064,821,387 バイト、SHA-256 は `2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d` です。公開用表示・校正記録のID表現・manifest の変更と、温度・統計・モデル本体等の一致は[公開記録](validation/jev27b/release.md)で区別しています。
- Swift の型(`TypedDecisionBundle`、`TypedDecisionModel` など)、`decision.json` の項目、CLI の入出力の詳細は英語のページの [Input and files](jev27b.md#input-and-files) にあります。

## 評価の詳細

評価の段、定義、表の全体は英語のページにあります。

- [Reference stages](jev27b.md#reference-stages): 段 0(配布の基底)、0q(基底の量子化)、1(統合前のアダプター)、1′(統合した checkpoint)、2(統合した重みの量子化を float32 で計算)、3(Core AI の資産)。
- [Quality and calibration](jev27b.md#quality-and-calibration): Q・Cal・H の段、事前に固定した条件と定義、結果の表、校正済み bundle の作り方。
- [Japanese (JGLUE)](jev27b.md#japanese-jglue): 日本語の 2 変種の wrapper、Jdev の抽出と採用の規則、J の結果の表。
- [Performance](jev27b.md#performance): 固定計画、性能用の行 P、計測の区切り、94 W の AC 給電の結果の表と、先に行った 70 W の AC 給電とバッテリー駆動の結果の表。

## ライセンスと帰属

Qwen3.8-27B(Qwen)と JEV-27B(AutoTrust AI)は Apache-2.0 です。JEV-27B は `SargeDev/jev-distill-corpus-v3`(Apache-2.0)で学習され、そのラベルの多くは TypeSafe Jev 1.13 の出力分布です。固定集合はこの corpus と JGLUE(CC BY-SA 4.0)から作り、再配布しません。Swift の試験の fixture には JGLUE の 8 行の本文が CC BY-SA 4.0 で入っており、その表示は `render.json` にあります。
