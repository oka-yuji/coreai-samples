# 言語生成ランタイムとSwiftからの利用

[トップ](../README.ja.md) · [English](runtime.md) · [検証手順](verification.ja.md)

ChatとWork、`coreai-chat` が使う言語生成ランタイムの開発者向けガイドです。Qwen3.8 27Bの導入、実測値、アプリ固有の制限は[モデル別ガイド](qwen3.8-27b.ja.md)を参照してください。コマンドは指定がない限りリポジトリのルートから実行します。

アプリのビルドにはRust/Cargoも必要です。[共通の準備](../README.ja.md#はじめる)と[IrodoriのMacデモ](irodori/demo.md)、[iPhone音声デモ](irodori/iphone.md)の確認状況を参照してください。

## 自分でバンドルを変換する

Chat と Work は、[apple/coreai-models](https://github.com/apple/coreai-models) の Apple の変換器で作ったほかの言語生成のバンドルも動かせます。このランタイムは Foundation Models の `LanguageModel` として動くので、アプリは通常の `LanguageModelSession` で話しかけます。

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

## 運用の注意

- **初回の読み込みでモデルを、計算ユニットの自動選択(優先する計算ユニットを指定せず、頻繁な reshape を期待する設定)で準備する。** 設定は `SpecializationOptions.default` に `expectFrequentReshapes = true` を足したものです。Core AI は準備済みのコピーを `~/Library/Caches/coreai-cache/<OS build>/<name>/` に置きます。`<name>` はコマンドラインのツールならプロセスの名前(`coreai-chat`、`coreai-verify`、`swift test` なら `swiftpm-testing-helper`)、アプリならバンドル ID(`com.coreai-samples.DemoApp`)です。同じプログラムの次の読み込みはそれを使い回します。別のプログラムは自分のコピーを準備し、モデルとほぼ同じ容量のディスクを使います(量子化していないバンドルではもっと大きい)。`coreai-chat --stats` で、コピーを使い回したか(`cached`)と準備にかかった時間(`specialize_s`)が分かります。GPU 向けに準備していた以前の版から更新した後の初回の読み込みでは、準備がもう一度走ります(再特殊化)。以前の設定で準備したコピーは、容量不足などで OS が消すか、下の方法で準備済みのコピーを消す(そのプログラムのそのバンドルの新旧両方が消えます)まで残ります。

- **準備済みのコピーを消すには**、呼び出したプログラムのそのバンドルのコピーを消す `Engine.deleteSpecializationCache(bundle:)` を呼ぶか、デモアプリの Models の画面の **準備済みキャッシュを削除**(Clear Prepared Copy。画面の確認では `--clear-prepared`)を使うか、フォルダを消します。

ライブラリでは `LocalLanguageModel.ReloadPolicy(readInterval:footprintLimitBytes:)` を、モデルの初期化の `reloadPolicy:` に渡して設定します。間隔を 0 にすると周期の読み直しを、footprint の上限を 0 にするとメモリの条件を止めます。

既定は、読み込みを伴う依頼 10 回と、プロセスの footprint 18,000,000,000 バイト(10 進の 18 GB)の条件です。閾値は 2026 年 10 月 9 日に、128 GiB の M4 Max、macOS 27.0(26A428)、Xcode 27 で、Qwen3.8-27B の 4bit の動的形状の `gdnfp32` の変種を使って較正しました。2026-10-09 13:03:12 UTC に始めた `M4-h36-memory-v2` の実行(run ID `3cc72f3295ab`)は、30 プロンプト、10 ターンごとに新しい会話、文脈の上限 16,384、ターンあたり最大 64 トークンの生成、間隔 10、footprint の条件なしでした。30 ターンすべてのターン直後のその場の footprint の最大は 11,937,577,240 バイトで、1.5 倍して 10 進の GB に切り上げると 18,000,000,000 バイトです。遅らせた標本はこの計算に使っていません。これは計測から決めた読み直しの条件で、どの端末でも安全なメモリの予算ではありません。

## 制限

- 言語生成のランタイムは、変換器の macOS 向けの書き出し(`*_dynamic` のフォルダ)の Qwen3、Qwen3.5、Qwen3.8 で動かしました。上の停止のトークン、`<think>` のタグ、ツールの呼び出しの形は、それらのテンプレートが使うものです。ほかの系統のモデルにはさらに対応が要ることがあります。Clef は別の `DecisionModel` の API を使います。

- 言語生成のランタイムはテキストを受け付けます。テキストでない transcript のセグメントは `LanguageModelError.unsupportedTranscriptContent` を投げます。Recipes の写真の入力には Apple の別の端末上のモデルを使い、Clef のバンドル自体はテキストだけです。

- guided generation はありません。応答の schema のある依頼は `LanguageModelError.unsupportedCapability` を投げます。

- デモの文脈の上限は 16,384 トークンです。それを超える文脈は検証の範囲の外です。

- 配る先のバンドルとランタイムで同時のセッションの確認が通らない限り、1 つのモデルで同時に使うセッションは 1 つにします。

- 取り消しを確かめるのは、プロンプトを読む前と、生成したトークンの間です。長いプロンプトを読んでいる途中では確かめません。

- デコードは公式の `sequential` エンジンと同じく、1 トークンずつの同期です。公式の既定のエンジンは処理をパイプラインにして、特に小さいモデルで速くデコードします(`coreai-verify speed` で 3 つを並べて比べられます)。

- 別に配布している Clef のバンドル(取得にはアクセス権が必要)は Decide と Recipes で使います。ChatとWorkには別の言語生成のバンドルが要ります。Qwen3.5 と Qwen3.8 の計測に使った変換器の版は[自分でバンドルを変換する](#自分でバンドルを変換する)にリンクがあり、1.0.0 のタグに含まれない pull request のものです。

## リポジトリの構成

```
CoreAISamples.xcodeproj  デモアプリのプロジェクト。開いて Run を押す
CoreAISamples/           macOS の SwiftUI のデモアプリ: Models、Chat(Chat／Work、RAG つき)、Decide、Recipes、Extract
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

- **Models** は、この Mac にあるバンドルのフォルダを開いて読み込み、準備したコピーを消します。カタログにはQwen3.8 27B (4bit)が載っていますが、現在の配布先は非公開で匿名取得できないため、取得済みフォルダを開きます。カタログのメモリの値「~32 GiB peak RSS while loading」(日本語表示では「約32 GiB（読み込み時のピークRSS）」)は、[Qwenの計測](qwen3.8-27b.ja.md#主な数値)に記したMac でそのバンドルを読み込んだときのアプリのプロセスの常駐メモリのピークで、`/usr/bin/time -l` で測りました(準備済みのコピーから 31.51 GiB、48 回の中央値。初回 31.88 GiB)。最小のメモリ要件ではありません。
- **Chat** は同じ会話の中でChat／Workを切り替えます。「検索・資料を使う」は起動時OFFで、新しい検索や文書取得をせず、モデルの知識と会話履歴から回答します。ONにしてWeb／Docsを選ぶと、資料を先に取得して回答します。モードは変わらず、WorkではOFFでもメモを保存できます。取得した資料と`[S1]`などの出典IDも同じ会話に保存します。[設定と履歴の仕様](specs/conversation-search-access.md)を参照してください。
- **Work** は Foundation Models のツールのループで メモを保存します。Webの検索・ページ取得は「検索・資料を使う」とWebがONのときだけ許可します。`fetch_page` で開けるのは、この会話で利用者が書いたURL、`web_search` の結果のURL、検索結果や取得したページの本文にあるURLだけで、それ以外は接続の前に断ります。修復や続きなどアプリが作った文とモデルの回答は数えません。通信しない再生(cassette の `replay` と固定資料の RAG 評価)では判定しません。`web_search` の検索語に会話の内容が入ることは防げません([詳細](demo-app.md#chat-and-work))。指示は、`fetch_page` で開いたページだけを引用するよう求めます。**Citation guard**(日本語表示では「引用URLの確認」)の設定は既定で off で、on にすると、引用した URL をその課題で取得したページと照合し(由来の判定で断った URL は取得していない扱い)、合わなければ 1 回だけ直させます([詳細](demo-app.md#citation-guard))。
- **Decide** は Clef-Flash で文書を編集できる選択肢で採点し、文書を使い回し、人の確定とモデルの候補を分けて扱います。
- **Recipes** は、確認した冷蔵庫の在庫と台所の道具でレシピを照合します。合成した写真 5 枚とそのレシピの URL を同梱し、開発者向けの詳細に実際の判定のスコアと時間を表示します。
- **Extract** は、Apple と Microsoft の公式の商品説明か、貼り付けたテキストから、出典つきのスペックのカードを最大 4 枚作ります。候補を開くと、元の条件を確かめられます。複数の構成や、見つからなかった項目も、そのまま表示します。これまでのラベルと点検メモの実験は **Extraction lab**(日本語表示では「抽出の検証画面」)に残しており、その B3 の結果は変わりません。商品の抽出は別の用途です。[仕様と検証の状況](specs/product-comparison-demo.md)を参照してください。

デモアプリの Chat と Work が扱うのは Qwen3、Qwen3.5、Qwen3.8 のバンドルです。`CoreAISamples/Demos/ModelManager/BundleSupport.swift` の `BundleSupport.supportedModelIDPrefixes` に、`"google/gemma-"` のようなほかの系統の Hugging Face のモデル ID の先頭を足すと、アプリはその系統のバンドルを開けるようになりますが、動くようにはなりません。ランタイムが扱う停止のトークン、`<think>` のタグ、ツールの呼び出しの形は Qwen のまま(制限を参照)なので、そのようなバンドルは読み込めても崩れた文で答えることがあります。系統の確認はこの一覧だけです。バンドルのフォルダを開くか読み込むときに、アプリはバンドルの `metadata.json` の `source.hf_model_id` を読み、その項目が無いか、どの先頭でも始まらなければ、Models の画面か Chat に理由を出して断ります。バンドルがどう変換されたかは確かめません。

試すには、`CoreAISamples.xcodeproj` を開いて Run を押します。コマンドラインでのビルド、ヘッドレスの自己テスト、画面の確認、ログは [docs/demo-app.md](demo-app.md) にあります。
