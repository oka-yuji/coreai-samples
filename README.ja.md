# Core AI Samples

[English](README.md)

AppleのCore AIとFoundation Modelsでオープンモデルを動かすSwiftサンプル集です。ローカルでの**会話と調査**、**文書の判断と抽出**、**画像検索**、**音声**を、変換手順や検証記録とともに試せます。

下の一覧からモデルを選んでください。各ガイドに導入手順・対応端末・実測値・制限をまとめています。状態は変換バンドルの公開状況を表し、アプリへの組み込みとは分けて扱います。

## モデルとロードマップ

確認日: **2026-10-19**。**公開済み（Released）**は変換バンドルを一般に取得できる状態、**着手中（In progress）**は変換・検証・公開準備を進めている状態、**対応予定（Planned）**は合意済みで未着手の作業です。「調査」と記したものは実現性の確認予定で、リリースや端末対応の確約ではありません。

リンクしているHugging Faceの4件には現在**非公開のバンドル**があり、取得にはアクセス権が必要です。この確認時点で匿名取得できるものはありません。変換の完了やアプリ実装のマージだけでは公開済みとせず、確認した範囲を各ガイドとIssueに記載します。

| モデル・バンドル | 用途 | 状態 | ダウンロード | ガイド・進捗 |
|---|---|---|---|---|
| Qwen3.8 27B · 4ビット、テキスト | Chat・Work・`coreai-chat` · macOS | 着手中 · バンドル非公開 | [アクセス権が必要](https://huggingface.co/okayuji/Qwen3.8-27B-coreai-4bit) | [ChatとWork](docs/qwen3.8-27b.ja.md) |
| Clef-Flash 9B · uint4＋fp16ヘッド、テキスト | Recipes・Decide · macOS | 着手中 · バンドル非公開 | [アクセス権が必要](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide) | [RecipesとDecide](docs/clef-flash.ja.md) |
| GLiNER2.5-multi · FP32 | 商品情報の抽出・点検メモ | 着手中 · 公開準備／iOS採用 | [アクセス権が必要](https://huggingface.co/okayuji/gliner2.5-multi-coreai-fp32) | [商品抽出デモ](docs/specs/product-comparison-demo.md) · [#210](https://github.com/oka-yuji/coreai-samples/issues/210) |
| EmbeddingGemma 2 · FP16／INT4-INT8 PTQ | 画像検索・カメラで物探し | 着手中 · 公開準備／iOS採用 | [アクセス権が必要](https://huggingface.co/okayuji/embeddinggemma-2-coreai) | [検索ガイド（英語）](docs/embeddinggemma2.md) · [#210](https://github.com/oka-yuji/coreai-samples/issues/210) |
| Irodori · FP32 | 音声の生成・再生 | 着手中 · アプリ／実機の確認 | — | [Mac（英語）](docs/irodori/demo.md) · [iPhone（英語）](docs/irodori/iphone.md) · [#93](https://github.com/oka-yuji/coreai-samples/issues/93) |
| Qwen Image 2.1 Turbo · macOS FP32 | 画像の生成・編集 | 着手中 · Mac変換検証済み／モバイル予定 | — | [変換手順（英語）](Tools/Conversion/recipes/qwen_image21/README.md) · [モバイル計画 #222](https://github.com/oka-yuji/coreai-samples/issues/222) |
| Magpie TTS Multilingual v2607 | 音声生成 | 着手中 · 変換と検証 | — | [#165](https://github.com/oka-yuji/coreai-samples/issues/165) · [PR #166](https://github.com/oka-yuji/coreai-samples/pull/166) |
| Qwen3-TTS CustomVoice · 0.6B | 音声生成 | 着手中 · 変換と検証 | — | [#167](https://github.com/oka-yuji/coreai-samples/issues/167) · [PR #168](https://github.com/oka-yuji/coreai-samples/pull/168) |
| Qwen3-TTS CustomVoice · 1.7B | 音声生成 | 着手中 · 変換と検証 | — | [#169](https://github.com/oka-yuji/coreai-samples/issues/169) · [PR #170](https://github.com/oka-yuji/coreai-samples/pull/170) |
| TeleOCR | 文書OCR | 対応予定 · 調査 | — | [#178](https://github.com/oka-yuji/coreai-samples/issues/178) |
| MiniCPM-V 4.7 1B | 画像・動画の理解 | 対応予定 · 調査 | — | [#179](https://github.com/oka-yuji/coreai-samples/issues/179) |
| Depth Anything 3 Small | 深度・カメラ姿勢 | 対応予定 · 調査 | — | [#180](https://github.com/oka-yuji/coreai-samples/issues/180) |
| LTX 2.5 | 動画生成 | 対応予定 · 調査 | — | [#181](https://github.com/oka-yuji/coreai-samples/issues/181) |

iOSアプリは通常起動で**Models**を開き、5つのデモを各providerへ接続しています。モデル配布定義の採用は [#210](https://github.com/oka-yuji/coreai-samples/issues/210) で進行中で、mainへ反映されるまでは通常のModels経路で説明だけを表示し、取得・実行はできません。[統合状況](docs/validation/ios-models-integration-20261018.md)を参照してください。ローカルの診断実行とバンドルの一般公開は別の到達点です。

## はじめる

開発には**AppleシリコンのMac、Xcode 27、Metal Toolchain**が必要です。MacデモはmacOS 27、iPhoneデモは対応を確認したiOS 27の実機を使います。iOSシミュレータではCore AIを実行できません。モデルごとのメモリ要件と検証済み端末は各ガイドで確認してください。

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
```

音声入力アダプター用にRust/Cargoをインストールします。`CoreAISamples.xcodeproj` を開き、**CoreAISamples**と**My Mac**を選んで実行してください。このスキームはReleaseでビルドします。デバッグ用には**CoreAISamples-Debug**があります。モデルの取得とフォルダ選択は各ガイドに従ってください。重みはGitに含まず、非公開バンドルの取得にはアクセス権のあるHugging Faceアカウントが必要です。

iPhone向けのビルド・署名、別アプリの音声デモは[アプリの導入手順（英語）](docs/demo-app.md#iphone-app)を参照してください。ビルド成功だけで、すべてのモデルがその端末に対応しているとは判断しません。

## 開発者向けガイド

- [Swift APIと言語生成ランタイム](docs/runtime.ja.md): 自分で変換する手順、CLI、対応する依頼、キャッシュと文脈、リポジトリ構成。
- [検証と計測](docs/verification.ja.md): ビルド・テスト、ランタイム比較、結果の計測条件。
- [デモアプリ（英語）](docs/demo-app.md): ビルド、ヘッドレス検証、画面確認、ログ。
- [変換ツール（英語）](Tools/Conversion/README.md): 再現可能な変換レシピと保守手順。
- [モデル中心の仕様書](docs/specs/model-first-demo.html)、[iOSデザイン](docs/designs/model-first-ios.html)、[macOSデザイン](docs/designs/model-first-macos.html)、[機能画面デザイン](docs/designs/function-screens.html): 設計案と実装状況。HTMLはローカルのファイルをブラウザで開くと整形して読めます。

## 関連

- [coreml-llm-samples](https://github.com/oka-yuji/coreml-llm-samples) — Core ML版。macOS 26／iOS 26向け。

## ライセンス

コードはMITです。[LICENSE](LICENSE)を参照してください。モデルの重みにはそれぞれのライセンスが適用され、各ガイドとバンドルのライセンス・告知ファイルで確認できます。Qwen3.8 27BとClef-Flash 9Bは元モデルのApache License 2.0を引き継ぎ、各Hugging Faceリポジトリに `LICENSE` と `NOTICE.md` があります。
