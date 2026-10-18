# Clef-Flash 9BをCore AIで使う

[トップ](../README.ja.md) · [English](clef-flash.md) · [Hugging Faceからダウンロード](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide)

文書と、答えの候補を決めたチェックリストを渡すと、候補ごとの確率を返します。文書を保持したまま質問を編集して再判定できます。アプリでは **Decide** と **Recipes** がこのモデルを使います。

**対象:** 文書分類や編集可能な判断項目を、Mac上で動かしたいSwift開発者。自由文のチャット、写真の直接入力、校正済みの本番判断サービスは対象外です。

必要環境はApple silicon、macOS 27、Xcode 27とMetal Toolchainです。GPUで実行し、計測した機種はM4 Max 128 GBです。最低メモリ構成は未確認です。Recipesの写真認識には、Macで画像入力対応のApple Intelligenceが利用可能である必要があります。

アプリをビルドする前に、Rust/Cargoを含む[共通の準備](../README.ja.md#はじめる)を済ませてください。

## クイックスタート

冷蔵庫にあるもので、このレシピは作れるか。**Recipes** は冷蔵庫の写真を読み、提案された食材を利用者が確認してから、その在庫をレシピの URL と照らし合わせます。写真には Apple の端末上のモデルを、レシピの必要な食材と道具には **Core AI で動く Clef-Flash 9B** を、数量には Swift のコードを使います。**Decide** は同じ Clef のバンドルで、文書を編集できる質問で採点します。どちらもモデルの校正していない確率を表示します。

**別に配布している Clef のバンドルへのアクセス権が必要です。** 下の `hf download` は、そのリポジトリへのアクセス権がある Hugging Face のアカウントでだけ動きます。[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)を入れ、そのアカウントで `hf auth login` を実行してから、アプリとバンドルを取得します。バンドルを開くときは `decision.json` を含むフォルダを選びます。

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
hf download okayuji/clef-flash-9b-coreai-decide --local-dir ./models/clef-flash-9b-decide-v1
```

`CoreAISamples.xcodeproj` を開き、**CoreAISamples** スキームと **My Mac** を選び、実行の構成を **Release** にして実行します。右上の **EN / JP** で、Mac の 6 つの画面の表示がすべて切り替わります。

サイドバーでレシピの説明が付いた **Clef-Flash 9B** を開き、**Clef-Flashのフォルダを選択**(Open Clef-Flash bundle)を選び、`models/clef-flash-9b-decide-v1` を選びます。**サンプル → 卵3個の卵焼き**(Examples → Three-egg omelette)を選び、提案された卵の数量を `1 pcs` にして確定し、**レシピを判定**(Check recipe)を押します。レシピには卵が 3 個要るので、確定した在庫では 2 個足りません。同梱の 5 枚の写真とレシピの URL は [Recipes の手順](recipes-demo.md)にあります。写真と在庫は Mac から出ません。レシピの URL の取得にはインターネット接続が要ります。

サイドバーで文書チェックリストの説明が付いた **Clef-Flash 9B** を開き、同じバンドルを選んで、**サンプル → 保守契約**(Examples → Maintenance agreement)を選んで **条件を判定**(Run checklist)を押します。質問を編集してもう一度実行すると、文書を使い回します。確認の付け方、JSON の書き出し、Swift の API は [Decide の説明](decide.md)にあります。

ビルドしたアプリは、ウィンドウを開かずに記録済みの入力も実行できます。`DECIDE_APP` にアプリの実行ファイルを指定します。

```sh
COREAI_DECISION_BUNDLE="$PWD/models/clef-flash-9b-decide-v1" "$DECIDE_APP" --decide-json models/clef-flash-9b-decide-v1/examples/maintenance-request.json response.json
```

これは[検証記録](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/58e654c86485b29a47fef820226977742ff6868a/VERIFICATION.md)と同じJSON実行経路です。

## 記録済みの出力

[入力全文](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/58e654c86485b29a47fef820226977742ff6868a/examples/maintenance-request.json)は、架空の保守契約書と5問です。「契約期間の満了時に、合意をし直さず自動的に更新する定めがあるか。」に対する記録済みの回答オブジェクトは次のとおりです。

```json
{"type":"noul","noul":0.0107}
```

`noul` は `true` の確率です。[応答全文](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/58e654c86485b29a47fef820226977742ff6868a/examples/maintenance-response.json)には全候補の確率とlogitも含まれます。この5問の全体読み直しは入力2,645トークンで4.277018792秒、同じ文書で質問を変更した再利用実行は0.775898292秒で、文書の2,067トークンを再利用しました。

計測は2026-10-08、M4 Max 128 GB、macOS 27.0（26A428）、Xcode 27 / Swift 6.4、Releaseで、モデル処理を一つずつ実行しています。フォルダを選んでbundleを開く操作は時間の外です。依頼の中で必要になるruntimeの準備・再読込は含みます。

## 検証した範囲と上限

以下も同じ機種・OS・ツールを使い、2026-10-07〜08にGPU処理を直列で実行した記録です。出典のJSONは[検証まとめ](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/58e654c86485b29a47fef820226977742ff6868a/VERIFICATION.md)から読めます。

| 検査 | 結果 |
|---|---|
| 固定10文書を、同じuint4重みをPyTorch fp32で計算した参照と比較 | 確率差の最大0.016693930、40問すべてで1位の候補が一致 |
| 固定4問・8K文書の再利用20回 | p95 0.846601958秒 |
| 同じ8K文書で1問と12問を交互に実行 | runtime再読込を含め1回約1.5〜2.7秒 |
| 入力全体 | 質問・選択肢を含む描画後の上限9,216トークン |
| 質問の枠 | 1〜16問、1問あたり1〜8候補 |

1秒未満という結果は固定4問の条件です。このアプリ統合の比較では512トークン以下の帯を検査していません。同じ量子化重みとの一致は、一般の課題での正解率や確率の校正とは別の検査です。

## モデルとアプリの担当

配布物はuint4のテキスト本体、別ファイルのfp16判断ヘッド、元のBF16語彙ベクトル、tokenizerをまとめたものです。本体のgated-deltaの状態更新はfp32です。ヘッドが渡された候補に点数を付け、runtimeが質問ごとにsoftmaxを計算します。

Recipesの写真を読むのは、Appleのon-device `SystemLanguageModel.default` です。提案された食材を人が確認します。Clefはレシピのテキストを読み、食材の必要性や設備を分類します。確定した在庫との照合・数量比較はSwiftコードが行います。モデルのダウンロードとレシピURLの取得にはインターネット接続が必要ですが、写真と在庫をレシピサイトへ送信しません。

## 入力とファイル

[Swift API](decide.md#swift-api)の `DecisionModel` を使います。入力は `state` と順序を保つ `questions`、質問型は `choice`、`noul`、`score` です。描画は上流 `joint_schema_model.py` のrevision `17f0b0ad64efb65d273590632833508766b2aae6` に対応し、通常のQwenチャットテンプレートは使いません。

モデルpayloadは15ファイル、合計7,278,548,313バイトです。2026-10-09の公開準備で再確認しました。主な内容は `backbone/*.aimodel`、`backbone/lexical.safetensors`、`backbone/tokenizer/`、`head/*.aimodel`、`decision.json`、`SHA256SUMS`、`LICENSE`、`NOTICE.md` です。説明文書と記録済みの例は別に付属します。モデル版は `decision.json` のSHA-256で、`67f69d5bd401104fd594ba74d94622882341c138be6149132d68f9857def852d` です。

## 読込と再利用

フォルダ構成を保って使ってください。読込時にmanifestと `SHA256SUMS` を検査します。`SHA-256 mismatch: <file>.` はファイルが変更されたか取得が不完全なことを示します。初回はGPU用の準備が走るため、ダウンロード分に加えてCore AIの準備済みキャッシュ用の空き容量が必要です。実行ファイルやOS buildが変わると別の準備済みコピーが必要になることがあります。

Decide画面の文書再利用では質問用に1,024トークンを予約します。上限超過時は切り詰めずにエラーを返します。新しい文書や全体読み直しは古いcontextを無効にします。`This document context is no longer current.` が出たらcontextを作り直してください。質問の形を変える場合は、同梱runtimeの再読込処理を使います。

## 制約

- 確率は、与えた候補の間の未校正の値です。不明・記載なしの候補を用意し、元の文書を確認してください。
- 配布bundleに画像encoder・音声入力・自由文生成はありません。iOSとメモリの小さいMacは未検証です。
- 固定文書での検査は、任意の文書や実際の冷蔵庫写真での正解率を示しません。uint4量子化によって元モデルから出力が変わることもあります。
- RecipesはKikkomanの公開レシピと別の写真モデルを使います。ネットワーク障害やApple Intelligenceが利用できない状態はエラーとして表示します。
- 開発者向けデモです。記録した20回の処理から、無期限の連続動作や本番品質は保証しません。

## ライセンスと帰属

モデル資産はApache License 2.0です。Qwen/Qwen3.5-9Bから事後学習した [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6) の条項を引き継ぎ、[元のLICENSE](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/main/LICENSE)とAlibaba Cloudの著作権表示をそのまま保持しています。変更内容と変換ツールは [NOTICE.md](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide/blob/main/NOTICE.md) に記載しています。このリポジトリのコードは [MIT](../LICENSE) です。

変換者はYuji Okaです。CloudflareやAppleによる公式配布ではありません。各名称・商標は元モデルとプラットフォームを示すために使っており、推奨や承認を意味しません。
