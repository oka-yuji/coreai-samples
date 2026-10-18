# Qwen3.8 27B で Chat と Work

[モデル一覧](../README.ja.md#モデルとロードマップ) · [English](qwen3.8-27b.md)

Qwen3.8 27Bの4ビット・テキスト専用バンドルを使い、Macで会話、Web調査、資料検索、メモ保存を試します。計測記録の「Agent」はWorkのツール実行経路を指します。

要件: AppleシリコンのMac、macOS 27、Metal Toolchainを入れたXcode 27。ダウンロードに16.85 GB、読み込むプログラムごとにほぼ同じ容量の準備済みコピーが必要です。読み込み時のピークRSSは34.2 GB（31.9 GiB）でした。128 GiBのM4 Maxで計測し、最小メモリ構成は未確認です。iPhone・iPadはこのバンドルの対象外です。

**配布状況（2026-10-19）:** バンドルはHugging Faceにありますが、現在は非公開です。以下の取得にはアクセス権のあるアカウントでの `hf auth login` が必要です。アプリ内の匿名ダウンロードは利用できないため、取得済みフォルダを開いてください。

アプリをビルドする前に、Rust/Cargoを含む[共通の準備](../README.ja.md#はじめる)を済ませてください。

## クイックスタート

[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) を入れて、アプリとバンドル(12 ファイル、16.85 GB)を取得します。

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
hf download okayuji/Qwen3.8-27B-coreai-4bit --revision 8e24aebe036eae577d4a3bd019ba728a73358318 \
  --local-dir ./models/qwen3.8-27b-coreai-4bit
```

`CoreAISamples.xcodeproj` を開き、**CoreAISamples** スキームと **My Mac** を選んで実行します(このスキームは Release でビルドします)。**モデル管理**(Models)で **フォルダを開く…**(Open Folder…)を選び、`models/qwen3.8-27b-coreai-4bit` を選んで **読み込む**(Load)を押します。初回の読み込みは M4 Max で 86.87 秒かかり、そのうち 82.84 秒が GPU 向けの準備でした。2 回目以降は 3.6〜10.3 秒でした。会話のタイトルにはモデル名を表示します。**チャット**(Chat)モードで話しかけるか、**ワーク**(Work)モードでタスクやメモ保存を行います。Webで調べるときは会話設定の「検索・資料を使う」と「Web」をONにします。**モデル管理**のダウンロードは同じ固定revisionを使いますが、匿名取得が必要なため、配布先が非公開の間は利用できません。画面の表示は右上の **EN / JP** で切り替えられます。

コマンドラインでは、`CoreAISamplesKit` フォルダで次を実行します。

```sh
cd CoreAISamplesKit
swift build -c release
.build/release/coreai-chat --model ../models/qwen3.8-27b-coreai-4bit --max-tokens 2048 --stats \
  --prompt "What is the capital of France? Answer with one word."
```

モデルは答える前に考えます。思考は `[reasoning]` の下に、回答は `[response]` の下に出て、`--stats` は読み込みと各ターンの記録を標準エラーに書きます。

### バンドル

| バンドル | |
|---|---|
| リポジトリと revision | [`okayuji/Qwen3.8-27B-coreai-4bit`](https://huggingface.co/okayuji/Qwen3.8-27B-coreai-4bit) の `8e24aebe036eae577d4a3bd019ba728a73358318`(アプリの Catalog が固定している revision。tag は `v1.0.0`) |
| 大きさ | 12 ファイル、16,852,543,780 バイト(16.85 GB、15.70 GiB)。モデル本体の `main.mlirb` は 16,839,684,298 バイト |
| 元のモデル | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) の revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。テキストのみ(画像のエンコーダは含まない) |
| 変換 | Apple の変換器の [pull request #321](https://github.com/apple/coreai-models/pull/321) の commit `36e9350`(1.0.0 のタグには含まれない)。preset `qwen3_5_cfg`(重みは 4 ビット、計算は bfloat16)で、gated delta の状態の更新だけ fp32 で計算する。macOS 向けの動的形状の書き出し |
| トークナイザ | `tokenizer/tokenizer.json` は元の Qwen3.8-27B のファイルで、変換器が保存し直したものと差し替えた。計測に使った 2,257 件の文字列では、どちらのファイルでも Swift のトークン ID は同じだった |
| メモリ | 準備済みのコピーからの読み込みでプロセスの常駐メモリは最大 33.8 GB(31.5 GiB)、初回の読み込みで 34.2 GB(31.9 GiB)。Agent の各ステップの間の常駐は 16.0〜16.7 GiB |
| 対応環境 | Apple シリコンの macOS 27、GPU。iPhone と iPad は対象外 |

### 生成例

アプリのヘッドレスの実行(`--agent-e2e`)での Agent の課題です。思考は off、生きた Web、2026 年 10 月 12 日。この run は `--instructions cite-fetched --citation-guard on` で動かしました。アプリの既定は、同じ指示で引用ガードが off です。回答と、その下の 2 つの注記は出力そのままです。

```
Read https://developer.apple.com/documentation/createml and list three key points.
```

```
Here are three key points from Apple's Create ML documentation:

1. **Train custom ML models on your Mac with familiar tools.** Create ML lets you use Swift and macOS playgrounds to create and train machine learning models locally. You train a model by showing it representative samples (e.g., many images of dogs to recognize dogs), then test it on unseen data and evaluate performance before integrating it into your app via Core ML.

2. **Broad model types are supported.** The framework covers image models (image classifier, object detector, hand pose classifier), video models (action classifier, hand action classifier, style transfer), text models (text classifier, word tagger, word embedding, gazetteer), sound classifiers, motion/activity classifiers, and tabular models (classifiers, regressors, recommenders).

3. **It leverages Apple's built-in ML infrastructure.** Because Create ML builds on the machine learning infrastructure used in Apple products like Photos and Siri, your image classification and natural language models are smaller and train much faster. It also provides built-in tools for evaluating model accuracy (metrics for classifiers, regressors, word taggers, recommenders, and object detectors) and controlling training sessions (MLJob, MLTrainingSession, checkpoints).

**Platform availability:** iOS 15.0+, iPadOS 15.0+, Mac Catalyst 15.0+, macOS 10.14+, tvOS 16.0+, visionOS 1.0+.

Sources:
https://developer.apple.com/documentation/createml
--- note Citations
1/1 cited URLs were retrieved in this task · mean 8-gram overlap 0.38
--- note Citation guard
Cited sources fetched · 1/1 cited URLs were fetched in this task
```

エージェントは `fetch_page` を 1 回呼び(6.3 秒、5,993 文字)、2 ステップ目で答えました。課題全体は、準備済みのコピーからの 4.5 秒の読み込みのあと 33.3 秒でした。

| ステップ | プロンプトのトークン | 使い回し | 生成 | 最初のトークンまで | デコード |
|---|---|---|---|---|---|
| 1(ツールの呼び出し) | 877 | 0 | 34 | 7.97 秒 | 23.8 tok/s |
| 2(回答) | 2,189 | 911 | 317 | 4.95 秒 | 23.0 tok/s |

### 主な数値

MacBook Pro(Mac16,6)、M4 Max、128 GiB、macOS 27.0(26A428)、上のバンドル、greedy デコード、モデルのプロセスは 1 つずつ、2026 年 10 月 11〜12 日。ランタイムと Agent のコードはこのリポジトリと同じで、Agent の指示と引用ガードは表に示したフラグで指定しました。

| 計測 | 結果 | 条件 |
|---|---|---|
| デコードの速さ | 22.74 tok/s(生成 144 ステップの中央値) | Agent、凍結した 12 課題 × 4 周(48 run)、生きた Web、`--instructions frozen --citation-guard off`、思考 low(36 run)と off(12 run) |
| 最初のトークンまで | プロンプトの読み込みまで 5.54 秒、画面に最初の文字が出るまで 6.46 秒(同じステップの中央値) | 同じ 48 run |
| Agent の課題 | 凍結した採点で 48/48 合格、回答の品質も 48/48。意図しない描き直し 0、止まらなかった run 0 | 同じ 48 run |
| 引用ガード on の Agent | 42 課題のうち 41 を達成、1 が保留。ガードの判定は確認 35・修復 6・保留 1。デコード 22.90 tok/s、最初のトークンまで 5.33 秒 / 6.46 秒(151 ステップ) | 21 課題 × 思考 low と off、生きた Web、`--instructions cite-fetched --citation-guard on` |
| 読み込み | 初回は 86.87 秒(うち GPU 向けの準備 82.84 秒)、2 回目以降は 3.6〜10.3 秒(中央値 4.7 秒、48 run) | アプリ |
| 常駐メモリ | 準備済みのコピーからの読み込みでプロセスあたり最大 33.8 GB(31.5 GiB)、初回の読み込みで 34.2 GB(31.9 GiB)。Agent のステップの間は 16.0〜16.7 GiB | `/usr/bin/time -l` とアプリの `metrics.jsonl` |
| RAG | 画面に最初の文字が出るまで 15.53 秒(3 回の中央値)、資料 2,950 トークン、入力は全体で 3,120 トークン | アプリの `--selftest --rag`、資料の予算 3,000 トークン、ローカルの文書だけ |

Agent の既定の設定(cite-fetched の指示、ガード off)は、このバンドルでは計測していません。参照の実装との一致などの正しさの確認は[検証](#検証)、制約は[制限](#制限)にあります。

## 検証

次の確認は、[上記バンドル](#バンドル)のモデルファイルを、主な数値に記した環境で2026年10月7〜12日に行いました。

- **Apple のランタイムと同じ出力。** `coreai-verify compare --max-tokens 1024` で 4 つのプロンプト(日本語、英語、コード、長い生成)を流すと、対応する `prefill-function` の経路で、Apple の sequential エンジンと 43、41、293、1,024 の各ステップのトークンがすべて同じで、logits の差の最大は 0、非有限の値は 0 でした。両方のランタイムが同じ変換済みのバンドルを動かしたので、これはランタイムの確認で、変換の確認ではありません。
- **元の重みとの一致は、測る前に置いた基準に届かなかった。** 参照は、同じ 4 ビットの重みを PyTorch で fp32、CPU で計算したものです。使っていなかった 40 本の入力(日本語、英語、ヒンディー語、タイ語、コード。本文と、Agent と RAG のプロンプトの形に描画したチャット)、各 512 位置で、参照の 1 位のトークンを Core AI も 1 位にした位置は 20,480 位置の 99.21%、実際の次のトークンに付けた確率の差は平均 0.002435、99 パーセンタイル 0.02796、最大 0.778 でした。差が 0.03 を超えた位置は 182(0.8887%)で、そのうち 26 位置は 1 位のトークンも違いました。NaN や無限大の値はありませんでした。平均と 1 位の一致率は基準(0.01 以下、99% 以上)を満たし、99 パーセンタイル(0.02 以下)、0.03 を超える位置の割合(0.5% 以下で、その全位置で 1 位が同じ)、最大(0.0875 以下)は満たしませんでした。大きい差は、英語の本文と、Agent と RAG の形の描画の中の、写した URL、版の数字、メッセージを終える確率に集まり、日本語・ヒンディー語・タイ語の本文で 0.03 を超えたのは 9,216 位置のうち 1 位置でした。Core AI の確率は、プロンプトを呼び出しにどう切るかでも変わります。このうち 10 本を 512 トークン 1 回ではなく 128 トークンずつ読ませると、5,120 位置のうち 77 位置で 1 位のトークンが変わりました。外れた 31 の位置から greedy で生成を続けると、アシスタント自身のターンは筋が通っていましたが、貼り付けた資料の続きでは 4〜5 本が崩れました(日本語の中のラテン文字、壊れたコード、コードの途中でのメッセージの終わり)。
- **4 ビットの量子化の影響は、トークンの確率でだけ測った。** 以前から使っていた 5 本の入力(日本語・英語・ヒンディー語・タイ語の世界人権宣言と Swift のソース。各 512 位置、2026 年 10 月 7 日)で、元の bfloat16 のモデルを PyTorch で GPU 上で動かしたものを参照にしました。このバンドルのモデルファイルを Core AI で動かすと、2,560 位置の 91.48% で 1 位のトークンが同じで、実際の次のトークンの確率の差は平均 0.03806(最大 0.9990)でした。同じ 4 ビットの重みを PyTorch の bfloat16 で動かすと 91.17%、平均 0.03862、KL の平均 0.0739 でした。同じ入力で、Core AI と、同じ 4 ビットの重みを fp32 で動かした PyTorch の差は平均 0.0019 でした。これは合否の基準を置かない参考の値です。量子化の前後での課題の品質は比べていません。
- **課題の結果は別の確認。** [主な数値](#主な数値)の 48/48 の Agent の run とガード on の run は、課題を終えられたかを測ったもので、元のモデルとの数値の一致を示すものではありません。凍結した指示の 48 本の回答を、2 人の採点者(同じ AI のモデルと同じ手引き。独立した人ではない)が、エージェントが受け取ったテキストと照らして読み、裏づけ「ある」22、「一部」15、「ない」11、要求を満たした回答は 48 でした。「ない」11 のうち 10 は、ページを開かずに検索の抜粋に頼っていました。
- **状態とサンプリング。** このバンドルで確かめたツールのループの 9 run で処理済みの状態の確認はすべて合格し、seed を固定した top-k と top-p のサンプリングは 5 回中 5 回同じトークンを繰り返し、seed を変えると変わりました。

## 運用の注意

- **Qwen3.8 27B の初回の読み込み。** アプリは初回にバンドルを 82.84 秒で準備し、86.87 秒で読み込みを終えました。準備済みのコピーからの読み込みは 3.6〜10.3 秒でした(2026 年 10 月 11 日)。`coreai-chat` は 16.86 GB(15.70 GiB)の自分のコピーを作り、ほかのビルドで Mac が混んでいたとき(1 分の負荷平均 20〜29)に 105.6 秒かかりました。

- **ダウンロードが遅いとき。** `hf download` が 1 MB/s を下回ったままなら、hf-xet のダウンロードの同時接続を上げると、1 回の計測で速くなりました。`HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY=32 HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY=32` を付けると、このバンドルは 18.91 MB/s(891 秒)で取れ、既定の設定では 0.2〜0.5 MB/s でした(hf-xet 1.5.2、huggingface_hub 0.36.2、2026 年 10 月 11 日)。

- **長く使うとき。** Qwen3.8 27B で `coreai-verify chat` の新しい会話を 100 回続けると(各最大 256 トークン)、デコードの速さは最初の 10 回の 22.33〜23.31 tok/s から、41 回目以降は 20.11〜21.62 tok/s のままでした(読み直しの直後の最初の質問を除く)。常駐メモリは 10 回目から 100 回目で 0.28% 増え、10 回ごとの読み直しは 4.31〜4.77 秒でした。速さが落ちる原因は分かっていません。

## 制限

ランタイム共通の[入力・文脈・取消の制限](runtime.ja.md#制限)も適用されます。

- Agent のページの取得と Web の RAG は、名前を解決したすべてのアドレスが公開のものでなければ取得しません。ループバック、私的、リンクローカル、未指定、共有、マルチキャスト、予約のアドレスを、IPv4 を写した IPv6 や古い数値の IPv4 の書き方も含めて断ります。検査は、最初の依頼の前、URLSession のリダイレクトのたび、WebView の主フレームと副フレームのナビゲーションのたび、最後の URL で行い、WebView の content rules もサブリソースのローカルの名前と公開でない数値のアドレスを止めます。名前を解決できなければ取得を断ります。検査と接続の間の DNS rebinding と、私的なアドレスに解決される名前への JavaScript やサブリソースの依頼は、防げません。アプリは App Sandbox なしで動きます。

- このQwenバンドルとChat／Workの検証対象はAppleシリコンのmacOS 27.0です。ほかのモデルのiOS対応とは別に扱います。

- Qwen3.8 27B の変換は、測る前に置いた一致の基準を満たしませんでした([検証](#検証))。わずかな割合の位置で、Core AI の確率が PyTorch の同じ重みと違い、特に英語の本文と、チャットに描画したツールの結果や資料で違います。

- Agent の回答の下の **Citations** の注記は、引用した URL が課題の検索結果か取得したページに出ていれば「取得した」と数えます。そのため、ページを 1 つも開かなかった課題で `4/4 cited URLs were retrieved in this task` と出たことがあります。引用ガードは、代わりに `fetch_page` が渡したページと照合します。

- 引用ガード(既定は off)が確かめるのは、引用したページをその課題で取得したことで、回答がページの内容と合っていることではありません。取得に失敗した URL は件数だけで示します。URL の直後に空白なしで続く文字(日本語の文など)は、その URL の一部として読みます。

- ガードを on にした生きた Web の 42 run のうち 1 run が保留で終わりました。最初の回答は開いていないページを引用していて、修復のターンでそのページは取得できましたが、直した回答がトークンの上限で止まりました。そのためアプリは最初の回答と、そのページを未取得とする照合の結果を残し、その旨の注記を付けました。ガードを既定にするために事前に決めた確認は、このような run を未取得の引用として数え、1 つも許さないので、既定は off です。

- 広告などの埋め込みのフレームが Web 以外のアドレスへ移るページでは、ページの URL が正しくても `fetch_page` が `error: fetch_page needs an absolute http or https URL` で失敗することがあります(凍結したベンチマークの 48 run のうち 4 run)。

- Agent は、読んだページに書かれた指示に従うことがあります。これを試すために書いた 5 つのページでの 10 run(引用ガード on)のうち 2 run が、ページが求めたメモを保存しました。ページが求めた URL を取得した run、メモのフォルダの外に書いた run はありませんでした。ツールが書くのはメモのフォルダだけで、**ファイル操作を許可**(Allow file operations)を on にすると `move_note` が加わります。

- `move_note` はメモのフォルダも移動先として受け付けますが、モデルに見せる説明とエラーの文は Desktop、Documents、Downloads しか挙げていません。

- Chatの既定は1回答あたり16ステップ、Workは別設定で既定8ステップです。会話設定の「1回答あたりの最大ステップ数」で、選択中のモードを1〜64に調整できます。通常の質問ごとに上限を数え直します。上限に達すると、取得済み情報を使い、ツールと推論を無効にして回答を1回だけ追加生成します。停止操作と出力トークン上限は引き続き有効です。文脈不足や生成失敗の際は調査履歴と終了理由を残します。この経路はFoundation Modelsを使うCPU統合テストで検証しており、Qwen3.8 27Bでの回答品質は未計測です。[仕様と検証](specs/conversation-step-budget.md)を参照してください。

- Qwen3.8 27B で測っていないもの: Agent の既定の設定(cite-fetched の指示、ガード off)、4 ビットの量子化の前後での課題の品質、128 GiB の M4 Max 以外の Mac、macOS 27.0(26A428)以外のビルド、16,384 トークンを超える文脈。
