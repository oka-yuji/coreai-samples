# Core AI Samples

[English](README.md)

Apple の Core AI と Foundation Models で、オープンなモデルを Mac 上で動かす Swift のサンプルです。**Chat** と **Agent** は Qwen3.8 27B で動き、Agent は Web を検索し、ページを読み、メモを保存します。**Recipes** は冷蔵庫の写真の食材とレシピの URL を照らし合わせ、**Decide** は文書を編集できるチェックリストで採点します。この 2 つは Clef-Flash 9B を使います。**Extract** は GLiNER2.5-multi で商品説明から出典つきのスペックのカードを作ります。`Tools/Verify` は、このリポジトリのランタイムと Apple のランタイムをトークン単位で比べます。

デモの主タイトルは **Clef-Flash 9B**、**GLiNER2.5-multi**、**EmbeddingGemma 2**、**Irodori** に統一しています。会話には選択中または保存済みのモデル名を表示します。同じモデルの複数デモは、サイドバーの説明やiPhoneメニューのモデル別セクションで区別します。ChatとWorkは会話のモード名です。

## 対象読者

- **向いている**: Mac 上で 27B のチャットモデルと Web を閲覧するエージェントを `LanguageModelSession` からローカルに動かし、その裏づけの計測も確かめたい Swift 開発者。
- **向いている**: レシピの確認や文書の判定をローカルで試し、スコアを確かめ、コードを再利用したい Swift 開発者。
- **向いている**: Foundation Models を通じてオープンな言語生成モデルを組み込む開発者。Chat と Agent は、Recipes や Decide とは別のランタイムとバンドルを使います。
- **向いている**: そのようなランタイムを Apple のものと照合したい人。`Tools/Verify` が両者をトークン単位で比べます。
- **向いていない**: iPhone や iPad、デモの 16,384 トークンを超える文脈、Clef をチャットモデルとして使うこと、モデルの確率を、正しさを測った確率として扱うこと。

要件: Apple シリコンの Mac、macOS 27、Metal Toolchain コンポーネントを入れた Xcode 27。Qwen3.8 27B は、ダウンロードに 16.85 GB、読み込むプログラムごとにほぼ同じ大きさの準備済みのコピーが要り、読み込み中のプロセスの常駐メモリは 34.2 GB(31.9 GiB)に達しました。計測は 128 GiB の M4 Max で行い、必要な最小のメモリ構成は確かめていません。Recipes の写真の読み取りには、その Mac で画像に対応した Apple Intelligence も要ります。Clef は 128 GiB の M4 Max で計測し、必要な最小のメモリ構成は確かめていません。Clef と Recipes のデモは iOS で確かめていません。

## モデル

| バンドル | 使う画面・ツール | ダウンロード | 説明 |
|---|---|---|---|
| Qwen3.8 27B、4 ビット、テキストのみ | Chat、Agent、`coreai-chat` | [Hugging Face](https://huggingface.co/okayuji/Qwen3.8-27B-coreai-4bit) | 次の節のクイックスタート |
| Clef-Flash 9B、テキストのみ、uint4 のバックボーンと fp16 のヘッド | Recipes、Decide | [Hugging Face](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide)(アクセス権が必要) | [English](docs/clef-flash.md) · [日本語](docs/clef-flash.ja.md) |

## クイックスタート: Qwen3.8 27B で Chat と Agent

[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) を入れて、アプリとバンドル(12 ファイル、16.85 GB)を取得します。

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
hf download okayuji/Qwen3.8-27B-coreai-4bit --revision 8e24aebe036eae577d4a3bd019ba728a73358318 \
  --local-dir ./models/qwen3.8-27b-coreai-4bit
```

`CoreAISamples.xcodeproj` を開き、**CoreAISamples** スキームと **My Mac** を選んで実行します(このスキームは Release でビルドします)。**モデル管理**(Models)で **フォルダを開く…**(Open Folder…)を選び、`models/qwen3.8-27b-coreai-4bit` を選んで **読み込む**(Load)を押します。初回の読み込みは M4 Max で 86.87 秒かかり、そのうち 82.84 秒が GPU 向けの準備でした。2 回目以降は 3.6〜10.3 秒でした。会話のタイトルにはモデル名を表示します。**チャット**(Chat)モードで話しかけるか、**ワーク**(Work)モードでタスクやメモ保存を行います。Webで調べるときは会話設定の「検索・資料を使う」と「Web」をONにします。**モデル管理**から同じ固定の revision を直接ダウンロードすることもできます。*Qwen3.8 27B (4bit)* の横の **ダウンロード**(Download)は、すべてのファイルを Hub の値と照合します。画面の表示は右上の **EN / JP** で切り替えられます。

コマンドラインでは、`CoreAISamplesKit` フォルダで次を実行します。

```sh
cd CoreAISamplesKit
swift build -c release
.build/release/coreai-chat --model ../models/qwen3.8-27b-coreai-4bit --max-tokens 2048 --stats \
  --prompt "What is the capital of France? Answer with one word."
```

モデルは答える前に考えます。思考は `[reasoning]` の下に、回答は `[response]` の下に出て、`--stats` は読み込みと各ターンの記録を標準エラーに書きます。

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

## クイックスタート: Recipes と Decide

冷蔵庫にあるもので、このレシピは作れるか。**Recipes** は冷蔵庫の写真を読み、提案された食材を利用者が確認してから、その在庫をレシピの URL と照らし合わせます。写真には Apple の端末上のモデルを、レシピの必要な食材と道具には **Core AI で動く Clef-Flash 9B** を、数量には Swift のコードを使います。**Decide** は同じ Clef のバンドルで、文書を編集できる質問で採点します。どちらもモデルの校正していない確率を表示します。

**別に配布している Clef のバンドルへのアクセス権が必要です。** 下の `hf download` は、そのリポジトリへのアクセス権がある Hugging Face のアカウントでだけ動きます。[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) を入れて、アプリと判定のバンドルを取得します。

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
hf download okayuji/clef-flash-9b-coreai-decide --local-dir ./models/clef-flash-9b-decide-v1
```

`CoreAISamples.xcodeproj` を開き、**CoreAISamples** スキームと **My Mac** を選び、実行の構成を **Release** にして実行します。右上の **EN / JP** で、Mac の 6 つの画面の表示がすべて切り替わります。

サイドバーでレシピの説明が付いた **Clef-Flash 9B** を開き、**Clef-Flashのフォルダを選択**(Open Clef-Flash bundle)を選び、`models/clef-flash-9b-decide-v1` を選びます。**サンプル → 卵3個の卵焼き**(Examples → Three-egg omelette)を選び、提案された卵の数量を `1 pcs` にして確定し、**レシピを判定**(Check recipe)を押します。レシピには卵が 3 個要るので、確定した在庫では 2 個足りません。同梱の 5 枚の写真とレシピの URL は [Recipes の手順](docs/recipes-demo.md)にあります。写真と在庫は Mac から出ません。レシピの URL の取得にはインターネット接続が要ります。

サイドバーで文書チェックリストの説明が付いた **Clef-Flash 9B** を開き、同じバンドルを選んで、**サンプル → 保守契約**(Examples → Maintenance agreement)を選んで **条件を判定**(Run checklist)を押します。質問を編集してもう一度実行すると、文書を使い回します。確認の付け方、JSON の書き出し、Swift の API は [Decide の説明](docs/decide.md)にあります。

## 自分でバンドルを変換する

Chat と Agent は、[apple/coreai-models](https://github.com/apple/coreai-models) の Apple の変換器で作ったほかの言語生成のバンドルも動かせます。このランタイムは Foundation Models の `LanguageModel` として動くので、アプリは通常の `LanguageModelSession` で話しかけます。

**1. バンドルを作る**。Apple の変換器の準備(`uv` を使う)と対応モデルは[変換器の models の README](https://github.com/apple/coreai-models/blob/1.0.0/models/README.md) にあります。ここで Qwen3-0.6B に使ったコマンドは次のとおりです。

```bash
git clone https://github.com/apple/coreai-models.git && cd coreai-models && git checkout 1.0.0
uv run coreai.llm.export Qwen/Qwen3-0.6B --output-dir ~/coreai-bundles
```

これで `~/coreai-bundles/qwen3_0_6b_4bit_dynamic` ができます。このランタイムは、そのバンドル、Qwen3.5-0.8B と Qwen3.5-9B、4 ビットと 8 ビットの Qwen3.8-27B で動かしました。Qwen3.5 と Qwen3.8 の書き出しには、1.0.0 のタグではなく [pull request #321](https://github.com/apple/coreai-models/pull/321) の版の変換器を使いました。

**2. チャットする**。`CoreAISamplesKit` フォルダで次を実行します。

```bash
cd CoreAISamplesKit
swift build -c release
.build/release/coreai-chat --model ~/coreai-bundles/qwen3_0_6b_4bit_dynamic --system "You are a concise assistant." \
  --prompt "My name is Taro. Remember it." --prompt "What is the capital of France? Answer with one word." \
  --prompt "What is my name?" --max-tokens 1024 --stats
```

`--prompt` の 1 つが、同じ会話の 1 ターンです。モデルが考えると、思考は `[reasoning]` の下に、回答は `[response]` の下に出ます。`--stats` は標準エラーに JSON の行を書きます。読み込みの 1 行(`cached`、`specialize_s`、`load_s`)と、ターンごとの 1 行です。ターンの行の `cached_tokens` は、前のターンから使い回したプロンプトの部分です。

**3. デモアプリを動かす**。Xcode で `CoreAISamples.xcodeproj` を開き、`CoreAISamples` スキームと `My Mac` を選んで Run を押します。初回は Xcode がパッケージの依存を解決します。`CoreAISamples` スキームは Release でビルドし、`CoreAISamples-Debug` はデバッグ用に Debug でビルドします。アプリでは **モデル管理**(Models)で、手順 1 のバンドルのフォルダを開きます。

## Swift から使う

```swift
import CoreAIBackend
import Foundation
import FoundationModels

let model = try await LocalLanguageModel(bundle: URL(filePath: "/path/to/qwen3_0_6b_4bit_dynamic"))
let session = LanguageModelSession(model: model, instructions: "You are a concise assistant.")
for try await snapshot in session.streamResponse(
    to: "What is the capital of France? Answer with one word.",
    options: GenerationOptions(maximumResponseTokens: 1024)
) {
    print(snapshot.content)
}
```

これは `CoreAISamplesKit/Sources/coreai-chat/ChatCommand.swift` の中心部分を、要る呼び出しだけに絞ったものです。各スナップショットには、その時点までの回答全体が入ります。CLI は `--prompt` ごとに 1 ターンを同じセッションへ送り、新しい部分だけを表示します。思考は `session.transcript` の `.reasoning` の項目に、トークン数は `session.usage` に、止まった理由は応答のメタデータ(`stopReason`)に入ります。`LocalLanguageModel(bundle:contextLimit:overflow:)` は文脈の上限と `OverflowPolicy`(後述)も受け取ります。デモアプリと `Tools/Verify` は、このリポジトリの `CoreAISamplesKit` パッケージの `CoreAIBackend` を、ローカルのパスの依存として使います。パッケージがリポジトリのフォルダの中にあるため、Xcode の Add Package Dependencies にこのリポジトリの URL を足す使い方には対応していません。自分のアプリで使うときは、リポジトリを clone し、`CoreAISamplesKit` フォルダをローカルのパッケージとして足して(File > Add Package Dependencies > Add Local)、`CoreAIBackend` をリンクします。更新は clone を通して取り込みます。

## ランタイムがすること

- **入力の文字列から制御トークンを作らせない。** バンドルの `added_tokens` にある文字列そのものを、利用者のプロンプト(RAG の資料を含む)とツールの結果から、描画の前に取り除きます。入れ子の記号や、取り除いた残りがつながってできる記号も扱います。システムの指示とモデルが生成した履歴は変えません。その文字列を含まないテキストはそのまま残ります。

- **読んだものを使い回す。** 各依頼を、エンジンがすでに持っているトークンと比べ、最初に違う位置より後ろだけを読み直します(混成モデルでは、いちばん近い保存地点から。次の項目)。会話のターンをまたいでも、ツールのループのステップをまたいでも同じです。使い回した数は、セッションの usage の `cachedTokenCount` に出ます。同じ `LocalLanguageModel` から作ったセッションは 1 つのエンジンを共有し、依頼は 1 本ずつ実行されます。
- **混成モデルでは保存地点に戻る。** 大きさの決まった再帰の状態を持つバンドル(Qwen3.5 と Qwen3.8)はキャッシュを切り詰めて巻き戻せないので、利用者のターンの終わり、生成の始まり、2 つのプロンプトが分かれる位置でその状態を保存し、先頭から読み直す代わりにいちばん近い地点を復元します(`Engine.checkpoint()`、`Engine.restore(_:)`)。
- **思考と回答を分ける。** `<think>` と `</think>` の間のテキストは `.reasoning` の項目として、残りは応答として流れます。`ContextOptions.reasoningLevel` はチャットテンプレートに `reasoning_effort` として届きます(`.custom("off")` は `enable_thinking` を false にします)。効くかどうかはテンプレートによります。
- **ツールを呼ぶ。** ツールの定義はチャットテンプレートに入り、呼び出しは、ストリーミングの途中で `<tool_call>` のブロックから、JSON(`{"name": ..., "arguments": ...}`)と `<function=...>` に `<parameter=...>` のどちらの形でも解析します。形の崩れたブロックはテキストとして回答に残ります。`toolCallingMode` が `.disallowed` のときは、ツールをプロンプトに入れません。ツールを実行してモデルにもう一度聞くのは Foundation Models です。
- **文脈が満杯のときは 2 つのどちらかで扱う**(`OverflowPolicy`)。既定の `.error` は、プロンプトそのものが入らなくなったときだけ `LanguageModelError.contextSizeExceeded` を投げ、そうでなければ残りの分を応答に使います。`.dropOldToolOutputs` は、応答の余地を空けるため、まず古いツールの結果を `[truncated]` に置き換え(新しい 2 つのまとまりは残す)、次に新しいものを半分にします。それでも足りなければ `.error` に戻ります。
- **指定どおりにサンプリングする。** サンプリングの指定が無ければ greedy でデコードします。`GenerationOptions` のサンプリングのモード `.random(top:seed:)` と `.random(probabilityThreshold:seed:)`、それに `temperature` に従い、同じ seed なら同じテキストになります。
- **きれいに止まり、取り消せる。** 生成は終端のトークンか `<|endoftext|>`(停止のトークンはモデルに戻さない)、または `maximumResponseTokens` で終わり、応答のメタデータがどちらかを示します(`stopReason`: `stopToken` または `maxTokens`)。考えている途中でトークンが尽きた応答は、エラーではなく空の回答で終わります。取り消した依頼のあとも、同じセッションの次の依頼は動きます。

ランタイムは、読み込みを伴う依頼が 10 回終わると、次に新しい会話が始まるときに読み直します。prefill が始まった依頼は、失敗や取り消しで終わっても数えます。読み込みの無い使い回しは数えません。Agent のツールのステップと、その後のターンは同じ会話のままです。プロセスの footprint の上限を設定すれば、会話の途中でも次の読み込みの前に読み直せます。

読み直しは、エンジンと保存地点を解放し、CPU 側の履歴は残し、プロンプト全体を新しいエンジンに読ませます。`sharedTokens` は論理的な共通部分を保ち、キャッシュした入力は 0 になり、prefill はプロンプト全体を読みます。置き換えを読み込んだ依頼だけが `reloadSeconds` を記録します。読み直しはモデルの読み込みの分だけ時間が増え、会話の途中なら、さらにプロンプト全体の prefill が加わります。下の較正の実行では、周期の読み直しは 27B で 4.19〜4.23 秒、0.8B で 0.145〜0.153 秒でした。15,797 トークンの 27B の会話の途中で強制した読み直しは、読み直しとプロンプト全体の prefill を含めて、最初の応答の項目まで 92.23〜92.89 秒でした。長い履歴ほど、再開に時間がかかります。

## 対応する依頼

executor は、推論の状態を取得・変更する前に依頼を検査します。表は macOS 27.0 の SDK の公開されている依頼の項目を扱います。生成を決めるのは今の依頼です。以前のプロンプトに保存された設定は、その時のターンを表すだけで、新しい指示として再び適用しません。

| 項目または場合 | 振る舞い |
|---|---|
| `LanguageModelExecutorGenerationRequest.id` | 依頼の識別子。描画とサンプリングを変えない。 |
| `transcript` | テキストの指示、プロンプト、思考、応答、ツールの呼び出しと結果を描画する。知らない種類の項目は断る。 |
| `enabledToolDefinitions` | ツールの呼び出しが許されていれば描画し、許されていなければ省く。 |
| `schema` | `unsupportedCapability(.guidedGeneration)` で断る。プレーンテキストで依頼する。 |
| `generationOptions` | 下の項目がこの生成を決める。 |
| `contextOptions` | 下の項目がこの描画を決める。 |
| `metadata` | 空でない独自のメタデータは `unsupportedGenerationGuide` で断る。モデル固有の設定の受け渡しは実装していない。エラーにはキーの名前だけが出て、値は出ない。 |
| `GenerationOptions.samplingMode` / 非推奨の `sampling` | 2 つの名前は同じ設定を指す。省略すると greedy。 |
| `.greedy` | 最大値のうち最初のものを選ぶ。 |
| `.random(top:seed:)` | top-k サンプリング。seed を使う。 |
| `.random(probabilityThreshold:seed:)` | top-p サンプリング。seed を使う。 |
| `temperature` | サンプラーに渡す。省略すると 1。0 以下なら greedy で選ぶ。 |
| `maximumResponseTokens` | 応答を、文脈の残りの範囲に抑える。 |
| `toolCallingMode` の省略 / `.allowed` | 有効なツールを示すが、呼び出しを強制しない。 |
| `.disallowed` | ツールをテンプレートから省く。 |
| `.required` | 呼び出しの強制には制約つきのデコードが要るため、`unsupportedCapability(.guidedGeneration)` で断る。 |
| 将来のサンプリングやツールの呼び出しの種類 | ほかの設定に落とさずに断る。 |
| `ContextOptions.includeSchemaInPrompt` | schema が無ければ効果なし。schema のある依頼は断る。 |
| `reasoningLevel` の省略 | テンプレートの既定を使う。 |
| `.light` / `.moderate` / `.deep` | テンプレートの値 `low` / `medium` / `xhigh` に対応させる。 |
| `.custom("off")` | `enable_thinking` を false にする。 |
| `.custom("low")` / `.custom("medium")` / `.custom("xhigh")` | 名前のある段階と同じテンプレートの値を使う。 |
| ほかの独自の思考の値 | `unsupportedCapability(.reasoning)` と断った値で断る。 |
| `Transcript.Entry.instructions` | システムのテキストとツールの定義を保つ。 |
| `.prompt` | 利用者のテキストを描画の前に無害化する。 |
| `.reasoning` / `.response` | モデルが生成した履歴を保つ。 |
| `.toolCalls` | 呼び出しと引数を記録された順に組み立て直す。 |
| `.toolOutput` | 文脈の切り詰めを当ててから、残ったテキストを無害化する。 |
| `Transcript.Segment.text` | 対応する。 |
| `.structure` / `.attachment` と将来のセグメントの種類 | 推論の前に `unsupportedTranscriptContent` で断る。 |
| `Transcript.Prompt.id` | 保存された項目を識別する。モデルへの入力ではない。 |
| `segments` | テキストだけを受け付ける。 |
| `options` / `contextOptions` / `metadata` | 保存されたターンの情報。今の依頼の対応する項目が優先する。 |
| `responseFormat` | 今のプロンプトでは断る。過去の形式は、新しいプレーンテキストのターンを縛らない。 |

思考の強さはバンドルによります。Qwen3.8 は 3 つの effort の値を使います。Qwen3.5 のテンプレートは思考の on / off を使い、それらの effort の値を区別しません。ランタイムは思考のトークンの予算を足しません。知らない名前を黙って渡さないよう、独自の思考の名前は文書にある 4 つの値に限っています。

## リポジトリの構成

```
CoreAISamples.xcodeproj  デモアプリのプロジェクト。開いて Run を押す
CoreAISamples/           macOS の SwiftUI のデモアプリ: Models、Chat(RAG つき)、Agent、Decide、Recipes、Extract
CoreAISamplesKit/        アプリが使う Swift パッケージ
  Sources/CoreAIBackend/ ランタイム: Engine(Core AI の呼び出し)、LocalLanguageModel(Foundation Models)、ModelBundle(metadata.json)
  Sources/RAGContext/    取得した資料から答えるための段落、順位、トークンの予算、プロンプト、日付。文書フォルダの読み取り
  Sources/coreai-chat/   コマンドラインのチャット(ChatCommand.swift)
  Tests/                 CoreAIBackendTests と RAGContextTests(swift test)
Config/                  アプリの署名の設定: Signing.xcconfig と Local.xcconfig.template
project.yml              CoreAISamples.xcodeproj の XcodeGen の定義
Tools/Verify/            別のパッケージ: このランタイムを apple/coreai-models 1.0.0 と比べる coreai-verify
docs/demo-app.md         デモアプリの説明: ビルド、ヘッドレスの自己テスト、画面の確認、ログ
LICENSE                  コードの MIT ライセンス
```

## デモアプリ

`CoreAISamples` は、このリポジトリの `CoreAIBackend` と `RAGContext` で作った macOS の SwiftUI アプリで、公式の `coreai-models` パッケージは使いません。

- **Models** は、この Mac にあるバンドルのフォルダを開いて読み込み、GPU 向けに準備したコピーを消します。カタログには、ダウンロードできる Qwen3.8 27B (4bit) が載っています。カタログのメモリの値「~32 GiB peak RSS while loading」(日本語表示では「約32 GiB（読み込み時のピークRSS）」)は、下の計測の表の Mac でそのバンドルを読み込んだときのアプリのプロセスの常駐メモリのピークで、`/usr/bin/time -l` で測りました(準備済みのコピーから 31.51 GiB、48 回の中央値。初回 31.88 GiB)。最小のメモリ要件ではありません。
- **Chat** は同じ会話の中でChat／Workを切り替えます。「検索・資料を使う」は起動時OFFで、新しい検索や文書取得をせず、モデルの知識と会話履歴から回答します。ONにしてWeb／Docsを選ぶと、資料を先に取得して回答します。モードは変わらず、WorkではOFFでもメモを保存できます。取得した資料と`[S1]`などの出典IDも同じ会話に保存します。[設定と履歴の仕様](docs/specs/conversation-search-access.md)を参照してください。
- **Work** は Foundation Models のツールのループで メモを保存します。Webの検索・ページ取得は「検索・資料を使う」とWebがONのときだけ許可します。指示は、`fetch_page` で開いたページだけを引用するよう求めます。**Citation guard**(日本語表示では「引用URLの確認」)の設定は既定で off で、on にすると、引用した URL をその課題で取得したページと照合し、合わなければ 1 回だけ直させます([詳細](docs/demo-app.md#citation-guard))。
- **Decide** は Clef-Flash で文書を編集できる選択肢で採点し、文書を使い回し、人の確定とモデルの候補を分けて扱います。
- **Recipes** は、確認した冷蔵庫の在庫と台所の道具でレシピを照合します。合成した写真 5 枚とそのレシピの URL を同梱し、開発者向けの詳細に実際の判定のスコアと時間を表示します。
- **Extract** は、Apple と Microsoft の公式の商品説明か、貼り付けたテキストから、出典つきのスペックのカードを最大 4 枚作ります。候補を開くと、元の条件を確かめられます。複数の構成や、見つからなかった項目も、そのまま表示します。これまでのラベルと点検メモの実験は **Extraction lab**(日本語表示では「抽出の検証画面」)に残しており、その B3 の結果は変わりません。商品の抽出は別の用途です。[仕様と検証の状況](docs/specs/product-comparison-demo.md)を参照してください。

デモアプリの Chat と Work が扱うのは Qwen3、Qwen3.5、Qwen3.8 のバンドルです。`CoreAISamples/Demos/ModelManager/BundleSupport.swift` の `BundleSupport.supportedModelIDPrefixes` に、`"google/gemma-"` のようなほかの系統の Hugging Face のモデル ID の先頭を足すと、アプリはその系統のバンドルを開けるようになりますが、動くようにはなりません。ランタイムが扱う停止のトークン、`<think>` のタグ、ツールの呼び出しの形は Qwen のまま(制限を参照)なので、そのようなバンドルは読み込めても崩れた文で答えることがあります。系統の確認はこの一覧だけです。バンドルのフォルダを開くか読み込むときに、アプリはバンドルの `metadata.json` の `source.hf_model_id` を読み、その項目が無いか、どの先頭でも始まらなければ、Models の画面か Chat に理由を出して断ります。バンドルがどう変換されたかは確かめません。

試すには、`CoreAISamples.xcodeproj` を開いて Run を押します。コマンドラインでのビルド、ヘッドレスの自己テスト、画面の確認、ログは [docs/demo-app.md](docs/demo-app.md) にあります。

## 検証

ビルドとモデルを使わない確認の全体を、リポジトリのルートで実行します。Apple シリコンの Mac、macOS 27、`xcode-select` で選んだ Xcode 27 が要ります。

```bash
./check.sh
```

このスクリプトは、`CoreAISamplesKit` をビルドして試験し、`Tools/Verify` をビルドして CPU の試験を Release で実行し、macOS アプリを Release でビルドし、Agent のオフラインの自己テストを実行します。その前に、引き継いだ `COREAI_*`、`GLINER_*`、`DECIDE_*`、`CLEF_*` の設定を消すので、関係のないモデルの試験の設定で推論が始まることはありません。最後の行は、終了コード 0 の `CHECK PASS` か、終了コード 1 の `CHECK FAIL: <stage>` です。

言語モデルの確認を足すには、JSON の結果の検査のために Python 3 を入れ、バンドルのフォルダを `:` で区切って渡します。最初のバンドルは、公式のランタイムとの比較と状態の確認にも使います。Qwen3.5-0.8B のような小さいバンドルなら、この確認は短く済みます。

```bash
COREAI_TEST_BUNDLES=/path/to/bundle-a:/path/to/bundle-b ./check.sh
```

これで、モデルを使う試験、Apple の sequential エンジンと比べる 2 つの固定のプロンプト、自前のランタイムでのチャット 2 ターンとそのあとの処理済みの状態の確認が加わります。2 つのプロンプトは、対応する `prefill-function` の経路で、生成したすべてのトークンと各ステップの logits が一致しなければなりません。2 つの `main-only` の記録は prefill の経路が違うので診断用に残し、その一致は合否に使いません。記録の欠け、報告されたエラー、状態の確認の失敗は、確認のコマンドが正常に終わってもスクリプトを失敗させます。JSON の記録は `.build/check/` に保存されます。

ランタイムとデモアプリの計測は、次の環境で行いました。

| 構成要素 | 計測した構成 |
|---|---|
| macOS | 27.0、ビルド 26A428 |
| Xcode | 27.0、ビルド 27A266a |
| Mac | MacBook Pro(Mac16,6)、Apple M4 Max |
| ユニファイドメモリ | 128 GiB(137,438,953,472 バイト) |

`Tools/Verify` は別の Swift パッケージで、1.0.0 に固定した Apple の `coreai-models` パッケージに依存する唯一の場所です。ライブラリ、CLI、デモアプリは依存しません。その `coreai-verify` は、Apple のランタイム(`--runtime official`、`--engine default` か `sequential`)とこのランタイム(`--runtime own`)を動かし、計測ごとに JSON を 1 行出します。表の「own」はこのリポジトリのランタイムです。

| サブコマンド | すること |
|---|---|
| `chat` | 1 つの LanguageModelSession で複数のターンをストリーミングする。 |
| `tools` | ツールが要るかもしれない質問をし、モデルのステップごとと質問ごとに記録を出す。 |
| `bench` | 読み直しの時間を測り、そのあと固定のプロンプトを、毎回新しいセッションで 3 回ストリーミングする。 |
| `compare` | 自前のエンジンと公式の sequential エンジンに同じトークンを入れ、greedy のトークンと logits を比べる。 |
| `rewind` | 共通の前置きのあとに地点を保存して生成し、復元してもう一度生成する。比較のため、全体の読み直しの時間も測る。 |
| `speed` | 自前のエンジン、公式の sequential エンジン、公式の既定のエンジンを、1 つずつ別のプロセスで交互に動かす。 |

`chat`、`tools`、`bench` は `--runtime` と `--engine` を受け取り、`--runtime own` では `--context-limit` と `--drop-old-tool-outputs` も受け取ります。

```bash
cd Tools/Verify
swift build -c release
.build/release/coreai-verify compare --model <bundle> --max-tokens 64 --prompt "en=List three fruits, one per line."
.build/release/coreai-verify chat --model <bundle> --runtime own --max-tokens 64 --prompt "What is the capital of France? Answer with one word."
```

`CoreAISamplesKit` フォルダで `swift test` を実行すると、使い回しの計画、解析器、サンプラー、文脈の詰め方、RAG の段落と日付のような、モデルの要らない試験が流れます。モデルを読む試験は、`COREAI_TEST_BUNDLES` にバンドルのフォルダを `:` で区切って並べたときだけ流れます。

```bash
cd CoreAISamplesKit
swift test
COREAI_TEST_BUNDLES=/path/to/bundle-a:/path/to/bundle-b swift test
```

モデルを読む試験は 1 つの直列のスイートにまとめてあり、トークナイザだけの確認は独立しています。流す前に、`CLAUDE.md` にある重いジョブのロックを取ります。`seededSamplingRepeats` は、使い回しを止めた状態で、top-k(40)と top-p(0.95)、temperature 0.8、seed 7 の生成したトークン列全体を比べます。完全に再現するには、バンドル、プロンプト、サンプリングの設定、prefill の切り方が同じでなければなりません。`COREAI_SEED_REPETITIONS` は既定の 20 回を変えます。公開の候補の大きいバンドルには、同じ試験を 5 回で使います。

境目の試験は、16,384 トークンの文脈に対して、入力をちょうど 16,382、16,383、16,384、16,385 トークンで描画します。どちらの文脈あふれの方針でも、最初の 2 つは最大 2 トークンと 1 トークンの生成の余地を残し、残りの 2 つは上限と入力の数とともに `contextSizeExceeded` を投げなければなりません。別の履歴で、ツールの結果の切り詰め、そのメタデータ、縮められない文脈あふれを確かめます。状態の確認は、8 目盛りの同点近傍の規則を使って、全体の読み直しと比べます。

`twoSessionsShareOneModel` は 1 つのモデルに 50 組の依頼を送り、executor の呼び出しの重なりを記録し、相手のセッションの印が混ざらないかを確かめ、5 組に 1 組は最初の断片が届いたあとで片方の依頼を取り消します。回復と最後の状態も確かめます。`COREAI_CONCURRENCY_PAIRS=1` で、Thread Sanitizer で流すための 1 組を選びます。これらは明示したときだけ流れる確認で、飛ばしたモデルの試験はバンドルの検証になりません。

```bash
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_SEED_REPETITIONS=5 swift test --filter ModelIntegrationTests.seededSamplingRepeats
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_CONCURRENCY_PAIRS=1 swift test --sanitize=thread --filter ModelIntegrationTests.twoSessionsShareOneModel
```

自前のランタイムの `chat` と `tools` は `--reload-interval` と `--reload-footprint-bytes` を受け取り、0 でそれぞれの条件を止めます。`chat --new-conversation-every 10` は 10 プロンプトごとに新しいセッションを始めます。自前のランタイムのチャットの記録には `generated_tokens` があり、停止のトークンを除いて、サンプリングした出力の ID をすべて含みます(応答の上限に達したときの最後の未処理の ID も含む)。失敗した CLI のターンには、この項目がありません。Verification SPI は、依頼だけの検査を通って実行に入った最後の依頼を返します。デコードの前は空、成功すれば全体、デコードの失敗や取り消しではその時点の途中のサンプルです。依頼だけの検査で断られた依頼は実行のロックの前で止まり、前のスナップショット、処理済みの状態、会話の追跡を変えません。executor が動く前にフレームワークが断った場合も、スナップショットは変わりません。ターンとツールのステップの記録には、読み直しがあれば `reload_seconds` が入り、メモリの記録には graphics footprint のバイト数(graphics footprint の ledger と、その圧縮の ledger の和)が入ります。`chat --check-state` は、読み直した各ターンと最後の状態を確かめます。

```bash
COREAI_TEST_BUNDLES=/path/to/bundle swift test --filter ModelIntegrationTests
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_TEST_RELOAD_INTERVAL=0 COREAI_TEST_RELOAD_FOOTPRINT_BYTES=1 swift test --filter ModelIntegrationTests.twoSessionsShareOneModel
```

閾値を低くした同時の依頼の実行では、少なくとも 1 回の読み直しが観測されなければなりません。CPU の方針の試験と偽のエンジンの試験に加えて、モデルを使う結果が要ります。

メモリの増え方を測るには、30 プロンプトの自前のランタイムの `chat` に `--memory-sample-turn 10 --memory-sample-turn 20 --memory-sample-turn 30` を足します。既定では off です。選んだターンでは、まず元のその場の `turn` の記録を出し、そのあと同じモデルを保ったまま 5 秒間推論せずに待ってから、ターンの番号、`wait_seconds`、メモリを取るまでの実際の `elapsed_seconds`、いつものメモリの項目を持つ別の `memory-sample` の記録を出します。状態の確認を頼んでいれば、そのあとに行い、最後の確認も含みます。重複したターンの番号や存在しないターンの番号は、読み込みの前に断ります。

5 秒の間隔は計測の決まりで、どの端末でもメモリが落ち着いた、回収が終わったことを保証するものではありません。その場の値とピークも、遅らせた標本と並べて残します。footprint の閾値の較正は、引き続きターン直後のその場の footprint の最大を使い、1.5 倍して 10 進の GB に切り上げます。観測の項目と待ちと取得の振る舞いの CPU だけの確認は、`swift test --package-path Tools/Verify` で流れます。

### Qwen3.8 27B のバンドル

次の確認は、[モデル](#モデル)のバンドルのモデルファイルを上の環境で、2026 年 10 月 7〜12 日に行いました。

- **Apple のランタイムと同じ出力。** `coreai-verify compare --max-tokens 1024` で 4 つのプロンプト(日本語、英語、コード、長い生成)を流すと、対応する `prefill-function` の経路で、Apple の sequential エンジンと 43、41、293、1,024 の各ステップのトークンがすべて同じで、logits の差の最大は 0、非有限の値は 0 でした。両方のランタイムが同じ変換済みのバンドルを動かしたので、これはランタイムの確認で、変換の確認ではありません。
- **元の重みとの一致は、測る前に置いた基準に届かなかった。** 参照は、同じ 4 ビットの重みを PyTorch で fp32、CPU で計算したものです。使っていなかった 40 本の入力(日本語、英語、ヒンディー語、タイ語、コード。本文と、Agent と RAG のプロンプトの形に描画したチャット)、各 512 位置で、参照の 1 位のトークンを Core AI も 1 位にした位置は 20,480 位置の 99.21%、実際の次のトークンに付けた確率の差は平均 0.002435、99 パーセンタイル 0.02796、最大 0.778 でした。差が 0.03 を超えた位置は 182(0.8887%)で、そのうち 26 位置は 1 位のトークンも違いました。NaN や無限大の値はありませんでした。平均と 1 位の一致率は基準(0.01 以下、99% 以上)を満たし、99 パーセンタイル(0.02 以下)、0.03 を超える位置の割合(0.5% 以下で、その全位置で 1 位が同じ)、最大(0.0875 以下)は満たしませんでした。大きい差は、英語の本文と、Agent と RAG の形の描画の中の、写した URL、版の数字、メッセージを終える確率に集まり、日本語・ヒンディー語・タイ語の本文で 0.03 を超えたのは 9,216 位置のうち 1 位置でした。Core AI の確率は、プロンプトを呼び出しにどう切るかでも変わります。このうち 10 本を 512 トークン 1 回ではなく 128 トークンずつ読ませると、5,120 位置のうち 77 位置で 1 位のトークンが変わりました。外れた 31 の位置から greedy で生成を続けると、アシスタント自身のターンは筋が通っていましたが、貼り付けた資料の続きでは 4〜5 本が崩れました(日本語の中のラテン文字、壊れたコード、コードの途中でのメッセージの終わり)。
- **4 ビットの量子化の影響は、トークンの確率でだけ測った。** 以前から使っていた 5 本の入力(日本語・英語・ヒンディー語・タイ語の世界人権宣言と Swift のソース。各 512 位置、2026 年 10 月 7 日)で、元の bfloat16 のモデルを PyTorch で GPU 上で動かしたものを参照にしました。このバンドルのモデルファイルを Core AI で動かすと、2,560 位置の 91.48% で 1 位のトークンが同じで、実際の次のトークンの確率の差は平均 0.03806(最大 0.9990)でした。同じ 4 ビットの重みを PyTorch の bfloat16 で動かすと 91.17%、平均 0.03862、KL の平均 0.0739 でした。同じ入力で、Core AI と、同じ 4 ビットの重みを fp32 で動かした PyTorch の差は平均 0.0019 でした。これは合否の基準を置かない参考の値です。量子化の前後での課題の品質は比べていません。
- **課題の結果は別の確認。** [主な数値](#主な数値)の 48/48 の Agent の run とガード on の run は、課題を終えられたかを測ったもので、元のモデルとの数値の一致を示すものではありません。凍結した指示の 48 本の回答を、2 人の採点者(同じ AI のモデルと同じ手引き。独立した人ではない)が、エージェントが受け取ったテキストと照らして読み、裏づけ「ある」22、「一部」15、「ない」11、要求を満たした回答は 48 でした。「ない」11 のうち 10 は、ページを開かずに検索の抜粋に頼っていました。
- **状態とサンプリング。** このバンドルで確かめたツールのループの 9 run で処理済みの状態の確認はすべて合格し、seed を固定した top-k と top-p のサンプリングは 5 回中 5 回同じトークンを繰り返し、seed を変えると変わりました。

## 運用の注意

- **初回の読み込みでモデルを GPU 向けに準備する。** Core AI は準備済みのコピーを `~/Library/Caches/coreai-cache/<OS build>/<name>/` に置きます。`<name>` はコマンドラインのツールならプロセスの名前(`coreai-chat`、`coreai-verify`、`swift test` なら `swiftpm-testing-helper`)、アプリならバンドル ID(`com.coreai-samples.DemoApp`)です。同じプログラムの次の読み込みはそれを使い回します。別のプログラムは自分のコピーを準備し、モデルとほぼ同じ容量のディスクを使います(量子化していないバンドルではもっと大きい)。`coreai-chat --stats` で、コピーを使い回したか(`cached`)と準備にかかった時間(`specialize_s`)が分かります。
- **準備済みのコピーを消すには**、呼び出したプログラムのそのバンドルのコピーを消す `Engine.deleteSpecializationCache(bundle:)` を呼ぶか、デモアプリの Models の画面の **準備済みキャッシュを削除**(Clear Prepared Copy。画面の確認では `--clear-prepared`)を使うか、フォルダを消します。
- **Qwen3.8 27B の初回の読み込み。** アプリは初回にバンドルを 82.84 秒で準備し、86.87 秒で読み込みを終えました。準備済みのコピーからの読み込みは 3.6〜10.3 秒でした(2026 年 10 月 11 日)。`coreai-chat` は 16.86 GB(15.70 GiB)の自分のコピーを作り、ほかのビルドで Mac が混んでいたとき(1 分の負荷平均 20〜29)に 105.6 秒かかりました。
- **ダウンロードが遅いとき。** `hf download` が 1 MB/s を下回ったままなら、hf-xet のダウンロードの同時接続を上げると、1 回の計測で速くなりました。`HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY=32 HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY=32` を付けると、このバンドルは 18.91 MB/s(891 秒)で取れ、既定の設定では 0.2〜0.5 MB/s でした(hf-xet 1.5.2、huggingface_hub 0.36.2、2026 年 10 月 11 日)。
- **長く使うとき。** Qwen3.8 27B で `coreai-verify chat` の新しい会話を 100 回続けると(各最大 256 トークン)、デコードの速さは最初の 10 回の 22.33〜23.31 tok/s から、41 回目以降は 20.11〜21.62 tok/s のままでした(読み直しの直後の最初の質問を除く)。常駐メモリは 10 回目から 100 回目で 0.28% 増え、10 回ごとの読み直しは 4.31〜4.77 秒でした。速さが落ちる原因は分かっていません。

ライブラリでは `LocalLanguageModel.ReloadPolicy(readInterval:footprintLimitBytes:)` を、モデルの初期化の `reloadPolicy:` に渡して設定します。間隔を 0 にすると周期の読み直しを、footprint の上限を 0 にするとメモリの条件を止めます。

既定は、読み込みを伴う依頼 10 回と、プロセスの footprint 18,000,000,000 バイト(10 進の 18 GB)の条件です。閾値は 2026 年 10 月 9 日に、128 GiB の M4 Max、macOS 27.0(26A428)、Xcode 27 で、Qwen3.8-27B の 4bit の動的形状の `gdnfp32` の変種を使って較正しました。2026-10-09 13:03:12 UTC に始めた `M4-h36-memory-v2` の実行(run ID `3cc72f3295ab`)は、30 プロンプト、10 ターンごとに新しい会話、文脈の上限 16,384、ターンあたり最大 64 トークンの生成、間隔 10、footprint の条件なしでした。30 ターンすべてのターン直後のその場の footprint の最大は 11,937,577,240 バイトで、1.5 倍して 10 進の GB に切り上げると 18,000,000,000 バイトです。遅らせた標本はこの計算に使っていません。これは計測から決めた読み直しの条件で、どの端末でも安全なメモリの予算ではありません。

## 制限

- Agent のページの取得と Web の RAG は、名前を解決したすべてのアドレスが公開のものでなければ取得しません。ループバック、私的、リンクローカル、未指定、共有、マルチキャスト、予約のアドレスを、IPv4 を写した IPv6 や古い数値の IPv4 の書き方も含めて断ります。検査は、最初の依頼の前、URLSession のリダイレクトのたび、WebView の主フレームと副フレームのナビゲーションのたび、最後の URL で行い、WebView の content rules もサブリソースのローカルの名前と公開でない数値のアドレスを止めます。名前を解決できなければ取得を断ります。検査と接続の間の DNS rebinding と、私的なアドレスに解決される名前への JavaScript やサブリソースの依頼は、防げません。アプリは App Sandbox なしで動きます。

- macOS だけで動きます(Apple シリコン、macOS 27.0)。`CoreAISamplesKit/Package.swift` は iOS 27 を宣言していますが、iOS では何も動かしていません。デモアプリの iPhone のターゲット(`CoreAISamples-iOS`、Extract だけ。[docs/demo-app.md](docs/demo-app.md#extract-on-iphone) を参照)はビルドできますが、まだ実機で動かしていません。
- 言語生成のランタイムは、変換器の macOS 向けの書き出し(`*_dynamic` のフォルダ)の Qwen3、Qwen3.5、Qwen3.8 で動かしました。上の停止のトークン、`<think>` のタグ、ツールの呼び出しの形は、それらのテンプレートが使うものです。ほかの系統のモデルにはさらに対応が要ることがあります。Clef は別の `DecisionModel` の API を使います。
- 言語生成のランタイムはテキストを受け付けます。テキストでない transcript のセグメントは `LanguageModelError.unsupportedTranscriptContent` を投げます。Recipes の写真の入力には Apple の別の端末上のモデルを使い、Clef のバンドル自体はテキストだけです。
- guided generation はありません。応答の schema のある依頼は `LanguageModelError.unsupportedCapability` を投げます。
- デモの文脈の上限は 16,384 トークンです。それを超える文脈は検証の範囲の外です。
- 配る先のバンドルとランタイムで同時のセッションの確認が通らない限り、1 つのモデルで同時に使うセッションは 1 つにします。
- 取り消しを確かめるのは、プロンプトを読む前と、生成したトークンの間です。長いプロンプトを読んでいる途中では確かめません。
- デコードは公式の `sequential` エンジンと同じく、1 トークンずつの同期です。公式の既定のエンジンは処理をパイプラインにして、特に小さいモデルで速くデコードします(`coreai-verify speed` で 3 つを並べて比べられます)。
- 別に配布している Clef のバンドル(取得にはアクセス権が必要)は Decide と Recipes で使います。Chat と Agent には別の言語生成のバンドルが要ります。Qwen3.5 と Qwen3.8 の計測に使った変換器の版は[自分でバンドルを変換する](#自分でバンドルを変換する)にリンクがあり、1.0.0 のタグに含まれない pull request のものです。
- Qwen3.8 27B の変換は、測る前に置いた一致の基準を満たしませんでした([検証](#qwen38-27b-のバンドル))。わずかな割合の位置で、Core AI の確率が PyTorch の同じ重みと違い、特に英語の本文と、チャットに描画したツールの結果や資料で違います。
- Agent の回答の下の **Citations** の注記は、引用した URL が課題の検索結果か取得したページに出ていれば「取得した」と数えます。そのため、ページを 1 つも開かなかった課題で `4/4 cited URLs were retrieved in this task` と出たことがあります。引用ガードは、代わりに `fetch_page` が渡したページと照合します。
- 引用ガード(既定は off)が確かめるのは、引用したページをその課題で取得したことで、回答がページの内容と合っていることではありません。取得に失敗した URL は件数だけで示します。URL の直後に空白なしで続く文字(日本語の文など)は、その URL の一部として読みます。
- ガードを on にした生きた Web の 42 run のうち 1 run が保留で終わりました。最初の回答は開いていないページを引用していて、修復のターンでそのページは取得できましたが、直した回答がトークンの上限で止まりました。そのためアプリは最初の回答と、そのページを未取得とする照合の結果を残し、その旨の注記を付けました。ガードを既定にするために事前に決めた確認は、このような run を未取得の引用として数え、1 つも許さないので、既定は off です。
- 広告などの埋め込みのフレームが Web 以外のアドレスへ移るページでは、ページの URL が正しくても `fetch_page` が `error: fetch_page needs an absolute http or https URL` で失敗することがあります(凍結したベンチマークの 48 run のうち 4 run)。
- Agent は、読んだページに書かれた指示に従うことがあります。これを試すために書いた 5 つのページでの 10 run(引用ガード on)のうち 2 run が、ページが求めたメモを保存しました。ページが求めた URL を取得した run、メモのフォルダの外に書いた run はありませんでした。ツールが書くのはメモのフォルダだけで、**ファイル操作を許可**(Allow file operations)を on にすると `move_note` が加わります。
- `move_note` はメモのフォルダも移動先として受け付けますが、モデルに見せる説明とエラーの文は Desktop、Documents、Downloads しか挙げていません。
- Chatの既定は1回答あたり16ステップ、Workは別設定で既定8ステップです。会話設定の「1回答あたりの最大ステップ数」で、選択中のモードを1〜64に調整できます。通常の質問ごとに上限を数え直します。上限に達すると、取得済み情報を使い、ツールと推論を無効にして回答を1回だけ追加生成します。停止操作と出力トークン上限は引き続き有効です。文脈不足や生成失敗の際は調査履歴と終了理由を残します。この経路はFoundation Modelsを使うCPU統合テストで検証しており、Qwen3.8 27Bでの回答品質は未計測です。[仕様と検証](docs/specs/conversation-step-budget.md)を参照してください。
- Qwen3.8 27B で測っていないもの: Agent の既定の設定(cite-fetched の指示、ガード off)、4 ビットの量子化の前後での課題の品質、128 GiB の M4 Max 以外の Mac、macOS 27.0(26A428)以外のビルド、16,384 トークンを超える文脈。

## 関連

- [coreml-llm-samples](https://github.com/oka-yuji/coreml-llm-samples) — Core ML 版。macOS 26 / iOS 26 向け。

## ライセンス

コード: MIT。[LICENSE](LICENSE) を参照してください。モデルの重みは、それぞれのライセンスで別に配布しています。Qwen3.8 27B と Clef-Flash 9B のバンドルは元のモデルの Apache License 2.0 を引き継ぎ、Hugging Face の各リポジトリに `LICENSE` と、変更点を書いた `NOTICE.md` があります。
