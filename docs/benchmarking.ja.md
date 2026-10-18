# モデルの共通検証と記録

[English](benchmarking.md) · [モデル一覧](../README.ja.md#モデルとロードマップ)

プロトコル **1**、2026-10-19採用。モデルの変換版と測定条件を固定した記録から、READMEのmacOS/iOS表示と各モデルの表を生成します。検査ツールは記録形式と参照を確かめます。実機動作や品質の判断には、実際の検証とレビューが必要です。

## 表示する内容

すべてのモデルを「デモ対応」「性能と品質」「検証した範囲」「根拠と測定条件」「導入・制限・進捗」の順にそろえます。計画中のモデルにも同じ欄を置き、不足している記録と未計測の検証を区別して表示します。

| 表示 | 意味 |
|---|---|
| ✅ | 根拠に記した変換版・経路で、そのOSの実機アプリがモデルを実行した記録がある |
| △ | 実装済み、旧経路・診断経路の記録のみ、または実機確認待ち |
| — | そのOSのネイティブデモは未提供。変換CLIだけが動く場合も含む |
| 公開済み | 変換バンドルを一般に取得できる。デモ対応とは別 |
| 合格・不合格 | 条件と根拠を示した検証の結果 |
| この記録では未収録 | 共通記録に値がない。過去の根拠の移行漏れか、実際に未計測かを確認する |
| 未計測 | その範囲の結果がない。ゼロや合格として数えない |
| 対象外 | モデル・入力条件に適用しない理由を元の記録に明記した項目 |

Macでの変換成功だけではmacOSに✅を付けません。ビルド、シミュレータ、架空プレビュー、fakeによる試験も根拠にしません。旧iPhone診断の成功と、現行Models入口の動作は別です。✅の対象は詳細に記した変換版・操作だけで、本人受入、別機種・別OS、本番品質の検収は別に記録します。

## 選定した指標

MLPerf Clientは用途に応じた品質確認と、TTFT、トークン生成速度、課題全体の時間を扱います。この測定の考え方を参考に、以下をこのリポジトリの共通項目とします。[MLPerf Client](https://mlcommons.org/benchmarks/client/)

| 分類 | 記録するもの | 区別する条件 |
|---|---|---|
| 識別・入力 | 元モデルrevision、変換資産revision/hash、コードrevision、精度、tokenizer/前処理、機種/RAM、OS build、SDK/ツール、backend、入力/出力長、batch、並列数、評価データ版、seed、生成設定 | 比較に影響する条件を固定する |
| 準備 | 特殊化込みの初回ロード、特殊化済み・新プロセスでのロード、キャッシュ容量 | 初回準備と通常利用を分ける |
| 応答 | 要求・パイプライン完了時間、中央値・p95・件数、ストリーミングの初回出力 | 開始・終了点、host処理・同期を含む範囲を明記 |
| 資源 | 資産容量、特殊化後容量、RSSピーク、footprintピーク、推論中常駐、解放後の基準との差 | 指標・単位・生涯ピークか標本最大かを分ける |
| 品質 | 未使用の評価入力での課題達成率、用途別指標 | 変換の正しさとは別。失敗と分母を保持 |
| 参照との一致 | 元精度→量子化参照、同条件参照→Core AI、元モデル→最終出力 | 最大/平均誤差、正規化、cosine/top-1、事前に決めた許容値 |
| 継続利用 | 連続実行、取消・再試行、解放・再読込、不正入力、破損・取得中断、オフライン | 実際のアプリと資産を使う経路を確認 |

メモリは新規計測で生のバイト数を残し、RSSとfootprintを加算しません。必要に応じてシステム全体のmemory pressure・swap・wiredも記録します。ピーク値だけで端末の最低RAMを決めません。これは既存のCore AIガイドの計測規約を引き継ぎます。

| 用途 | 追加する指標 | 固定する条件 |
|---|---|---|
| LLM・VLM | TTFT、表示開始、思考後の回答開始（LLM）、tokens/s。必要ならITL | 入力/実出力token数、思考、tool、cache再利用、文脈長。VLMは画像/フレーム数と形状 |
| 判断 | 全文読込・文書再利用時の時間 | 文書長、質問/選択肢数。正解付き未使用文書、確率を解釈する場合の校正 |
| 抽出 | Precision、Recall、F1 | span一致規則、schema、文長、chunk、言語と評価集合 |
| 埋め込み・検索 | 索引/検索時間、Recall@k、nDCG@k、参照cosine | 件数、次元、k、関連度ラベル、閾値、前処理 |
| 音声生成 | 初回再生可能音声までの時間、RTF、CER、人の試聴評点 | 文章/言語/話者、seed、音声長/Hz、streaming、計測に含む工程 |
| 画像生成 | 完成画像までの時間、CLIP score、人の画質評価 | 解像度、step、scheduler、seed、guidance、参照画像、編集条件 |
| OCR | CER、WER、構造/レイアウトF1 | 正規化、読み順、言語、文書の種類、画像サイズ |
| 深度 | AbsRel、実スケールのRMSE、delta1 | 正解の単位、scale/shift合わせ、有効画素、評価集合 |
| 動画 | 完成までの時間、生成frames/s、人の時間的一貫性・内容評価 | frame数、FPS、解像度、step、seed、音声 |

TTFTは要求開始から最初の出力token/chunkまでです。思考を含む「最初の表示文字」や「最終回答の開始」と分け、`prefillSeconds`だけをTTFTへ読み替えません。decode速度とITLは最初のchunk以降の実token数と時間で定義し、chunk数とtoken数を混同しません。[NVIDIA AIPerfの定義](https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference)

検索の小さな固定例の合格率を、そのままRecall@kと呼びません。関連度と順位から計算する指標を別に記録します。[MTEB](https://arxiv.org/abs/2210.07316)

RTFは生成時間÷音声長です。前処理・ロード・保存を含むか明記します。これは速さの指標で、ASRによるCER/WERや人の自然さ評価とは別です。挿入誤りによりCER/WERは1を超える場合があるため、1以下に切りません。[Coquiの実装](https://github.com/coqui-ai/TTS/blob/dev/TTS/utils/synthesizer.py)、[CERの定義](https://huggingface.co/spaces/evaluate-metric/cer/blob/main/README.md)

CLIP scoreは画像と文章の対応を測る補助に使います。FID/KIDは正解側データ、標本数、計算実装を固定できる場合の追加指標とし、デモ数枚から一般画質のスコアを作りません。編集の保持や文字など、人の確認も別に残します。[Diffusersの評価ガイド](https://huggingface.co/docs/diffusers/main/conceptual/evaluation)

深度のAbsRel/RMSE/delta1には正解のスケールと有効画素の定義が必要です。[Depth Anythingの評価実装](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/util/metric.py)

エネルギー/要求・tokenと連続負荷時の熱・速度は、測れる道具がある場合の追加項目です。エネルギー未計測でも電源・低電力設定、thermal state、継続時間、他の負荷を記録します。新しい指標は定義・単位・検査を登録してから使用します。

## 計測から更新まで

1. 変換版、OS/機種、アプリ/CLIの経路を選び、主指標・品質基準・評価データ版・入力/出力形状・seed/設定・計測区間を**測る前に固定**します。校正と未使用の評価入力を分け、閾値の変更は計画の新版にします。
2. 計画テンプレートを出し、全項目を埋め、版/hashを固定します。

   ```sh
   mkdir -p .local
   python3 -m Tools.ModelReports template --kind llm --model qwen38-27b > .local/qwen-plan.json
   shasum -a 256 .local/qwen-plan.json
   ```

3. モデルを読む前に `CLAUDE.md` の共通heavy-job lockを取ります。基準計測はRelease・モデル1プロセス・並列1とし、電源、熱、GPUの他負荷、特殊化cacheの状態、ツール版を記録します。比較は同条件で交互に流します。
4. 初回特殊化、新プロセスのcache利用ロード、初回推論、定常推論を分けます。このプロジェクトの初期設定は固定条件ごとにwarmup 1回＋計測20回です。warmupを増やす場合は理由と実回数を残します。連続利用と解放・再試行も別に確かめます。過去の記録へ架空の反復数を足しません。
5. 既存の実行・検証入口でrunごとのJSONLを保存します。各行は `run_id`、`context`、`status`（`warmup` / `passed` / `failed`）、`metrics`。model・variant・device・OS・workload・cache・固定計画の `plan_sha256` が全行で一致し、各指標は数値 `value` と登録済み `unit` を持ちます。この形式と集計器自体はモデルを実行しません。[形式例](model-reports/run-format.example.jsonl)の値は合成データで、モデルの結果に移しません。
6. 記録を共通集計します。

   ```sh
   python3 -m Tools.ModelReports summarize --input .local/qwen-runs.jsonl > .local/qwen-summary.json
   ```

   成功・失敗・warmupの件数を分け、時間の統計は成功した計測runから計算します。失敗件数を消しません。中央値は中央の値（偶数なら中央2件の平均）、p95はnearest rankの `ceil(0.95*n)` です。成功標本20件未満ならp95は `null`。この件数は表示のためのプロジェクト内規則で、統計的な精度保証ではありません。末尾の遅延を主張するには目的に合う件数・不確実性の評価が必要です。p95を最悪値と呼ばず、最大と生データも保持します。
7. 生の根拠、用途品質、数値一致、運用確認を分けてレビューします。公開記録から認証情報、個人入力、取得ページ全文、ローカル絶対パス、端末識別子を除きます。変換reportは先に `python3 -m Tools.Conversion.summary` で公開可能な要約を取れますが、その検証stageの時間を推論時間へ転用しません。
8. モデル記録を更新し、生成・検査・レビューへ進みます。モデル、精度、入力、runtime、OSが変わったら旧条件と旧結果を残します。出荷可否は既存Core AIガイドの10項目で判断し、この表は根拠への入口に使います。

## 記録と追加方法

`model-reports/catalog.json` が順序と確認日、`model-reports/metrics.json` が項目名・単位・用途別の欄、`model-reports/records/<id>.json` が各モデルの結果の正本です。これらは文書用で、アプリのダウンロードカタログとは別です。

evidenceには根拠の場所、`native_app` / `diagnostic` / `conversion` / `plan`、モデルを実際に動かしたか、結果、条件を記録します。`standard` は全条件と固定revisionを必須にします。日付は元の結果・記録の日付とし、複数日の結果では期間をworkloadにも残します。未実施の計画・状況確認の日付は実行日と区別して注記します。`legacy` は元記録にない項目を `null` のままにし、不足項目を表へ出します。推測で埋めて新プロトコルの計測へ格上げしません。

測定値は指標名、値/単位、集計法、件数、evidence ID、日英の条件注記を持ちます。`source_value` にローカルJSONのpathとpointerを設定すると、検査時に値を照合できます。RTFなどの算出値は式と元の量を注記します。比較できない入力は別の行にし、数値一致が通っても品質の不合格を残します。

モデル追加時は `model-reports/records/teleocr.json` などの計画中の記録をコピーし、ID・kind・名前・用途・要約・公開状態の説明・pages・linksを変更して `model-reports/catalog.json` にIDを足します。実測値なし・確認未実施から始めます。日英のMarkdownに `<!-- model-report:start -->` / `<!-- model-report:end -->` を1組ずつ置き、その外に導入文を加えます。✅へ変える前に実機の根拠を記録します。既存の詳しい手順と過去の記録は生成ブロックの外に残します。

```sh
python3 -m Tools.ModelReports render
python3 -m Tools.ModelReports check
python3 -m unittest discover -s Tools/ModelReports/tests -v
```

`render` は指定ブロックだけを更新します。`check` は古い生成表、ローカルのリンク切れ、不正な数値・単位、根拠不足、計画/ビルド/診断のみを根拠にした✅を拒否します。未計測項目を明示した記録は通ります。`check --require-complete <model-id>` は、全共通/用途別項目に新プロトコルの記録または理由付き対象外があり、各確認が合格または理由付き対象外であることも要求します。一つの変換版・機種・OS・固定計画で必要項目がそろうことを要求し、別の変換版の結果で不足を埋めません。これは**報告の充足検査**で、科学的妥当性や出荷を自動承認しません。

`./check.sh` は通常のビルドより前に記録検査と専用テストを実行します。モデル・検証を更新するPRでは、元の数値、失敗した基準、対象のデモ経路と、生成した日英表を合わせて確認してください。形式検査の成功後も、根拠の読解と表示のレビューは必要です。

task・data・sourceと数値を構造化する考え方はHubのモデルカードにも対応します。現ツールの出力先はリポジトリのMarkdownです。[Hugging Faceの評価メタデータ](https://huggingface.co/docs/hub/model-cards#evaluation-results)
