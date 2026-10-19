# EmbeddingGemma 2 · FP16／INT4-INT8 PTQ

[トップ](../../README.ja.md) · [English](../embeddinggemma2.md)

画像検索・カメラで物探し

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](../benchmarking.ja.md) · [元の記録](records/embeddinggemma2.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | ✅ | FP16のローカル画像検索UIの記録。PTQのカメラ受入を含まない。 | [E1](../validation/embeddinggemma2-ready-20261016.md) |
| ios | △ | 旧FP16診断は成功。現行Models経路は公開済みのFP16とPTQを取得・検査する（iPhone 17 Proの自己テスト、モデルは読み込まない）。同経路でのデモ実行とPTQの実機推論は確認待ち。 | [E2](../validation/embeddinggemma2-ready-20261016.md) · [E4](../validation/ios-models-integration-20261018.md#device-checks) · [E3](../validation/embeddinggemma2-int4-int8-20261018.md) · [E7](../validation/ios-models-integration-20261018.md#acquisition) |

### 性能と品質

| 指標 | 結果 | 集計・件数 | 条件と根拠 |
|---|---|---|---|
| バンドル・資産容量 | 913,677,333 B | single; n=1 | [E3](../validation/embeddinggemma2-int4-int8-20261018.md) · FP16バンドル。tokenizerと告知ファイルを含む。 |
| バンドル・資産容量 | 341,880,041 B | single; n=1 | [E3](../validation/embeddinggemma2-int4-int8-20261018.md) · PTQバンドル。tokenizerと告知ファイルを含む。 |
| 特殊化キャッシュ容量 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 初回ロード（特殊化を含む） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 特殊化済みキャッシュからのロード | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 要求・パイプライン完了時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| ピークRSS | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| ピークphysical footprint | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 推論中の常駐メモリ | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 解放後メモリの基準との差 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 課題達成率 | 0.833333333 ratio | aggregate; n=6 | [E1](../validation/embeddinggemma2-ready-20261016.md) · 未使用6件中5件の閾値判定。Recall@k/nDCGではない。固定閾値で正しい1位画像を隠した失敗を含む。 |
| 課題達成率 | 0.833333333 ratio | aggregate; n=6 | [E2](../validation/embeddinggemma2-ready-20261016.md) · 未使用6件中5件の閾値判定。Recall@k/nDCGではない。固定閾値で正しい1位画像を隠した失敗を含む。 |
| 参照との誤差（定義は根拠参照） | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 停止受付から完了までの時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 索引の構築時間 | 0.76569325 s | single; n=1 | [E1](../validation/embeddinggemma2-ready-20261016.md) · 画像5枚の索引構築1回。旧経路であり、Mac/iPhoneの制御比較ではない。 |
| 索引の構築時間 | 0.895115834 s | single; n=1 | [E2](../validation/embeddinggemma2-ready-20261016.md) · 画像5枚の索引構築1回。旧経路であり、Mac/iPhoneの制御比較ではない。 |
| 検索クエリ時間 | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| Recall@k | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| nDCG@k | この記録では未収録 | — | 共通記録に値がありません。再計測の前に過去の根拠も確認してください。 |
| 参照とのcosine類似度 | 0.999483553 cosine | min; n=20 | [E1](../validation/embeddinggemma2-ready-20261016.md) · 元FP32と生テキスト・画像20入力を比較。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 合格（記載範囲） | FP16と同一PTQの数値契約のみ合格。PTQと元精度の差は別評価。 | [E1](../validation/embeddinggemma2-ready-20261016.md) · [E3](../validation/embeddinggemma2-int4-int8-20261018.md) |
| 用途の品質 | 不合格（記載基準） | 未使用入力の偽陰性を保持。PTQでは元の例も失敗し、合格のために閾値を下げていない。 | [E5](../validation/embeddinggemma2-ready-20261016.md) · [E6](../validation/embeddinggemma2-int4-int8-20261018.md) |
| 連続利用・資源解放 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 停止・再試行・回復 | 合格（記載範囲） | 旧画像索引経路の停止・待機後再試行。現行Modelsやカメラ割り込みの確認ではない。 | [E1](../validation/embeddinggemma2-ready-20261016.md) · [E2](../validation/embeddinggemma2-ready-20261016.md) |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](../validation/embeddinggemma2-ready-20261016.md) | 2026-10-16 / legacy / native_app | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 冷蔵庫画像5枚・未使用クエリ6件、768次元、閾値0.67。ローカルフォルダの画像検索UI。 · variant=safe FP16 S128 / vision70; backend=Core AI; GPU preferred; route=macOS local-folder image search; source_revision=914f7f89142e33e77833254d9c9b90c3cef7303b; batch_size=1; concurrency=1; 未記録: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E2](../validation/embeddinggemma2-ready-20261016.md) | 2026-10-16 / legacy / diagnostic | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | 旧署名アプリの診断経路：画像5枚・未使用6クエリ。現行Models入口とは別。 · variant=safe FP16 S128 / vision70; backend=Core AI; GPU preferred; route=diagnostic; batch_size=1; concurrency=1; 未記録: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E3](../validation/embeddinggemma2-int4-int8-20261018.md) | 2026-10-18 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 同じPTQの20ベクトル照合。元精度からの品質劣化を保持。iPhoneはロックで実行できなかった。 · variant=INT4 text / INT8 vision PTQ; backend=Core AI; GPU preferred; route=conversion; source_revision=914f7f89142e33e77833254d9c9b90c3cef7303b; artifact_revision=7c37ea823af7a4028ec3757e124b53fe8e16a75c99c46d24ca0a7007cca55b85; code_revision=71fb61afd24e1fc5324f57a1abc02744d07ae09c; batch_size=1; concurrency=1; 未記録: warmups, power, thermal, plan_sha256 |
| [E4](../validation/ios-models-integration-20261018.md#device-checks) | 2026-10-18 / legacy / plan | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | 根拠に記載された固定入力を参照。 · variant=FP16 / PTQ; backend=Core AI; GPU preferred; route=plan; batch_size=1; concurrency=1; 未記録: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E5](../validation/embeddinggemma2-ready-20261016.md) | 2026-10-16 / legacy / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 冷蔵庫画像5枚・未使用クエリ6件、768次元、閾値0.67。ローカルフォルダの画像検索UI。 · variant=safe FP16 S128 / vision70; backend=Core AI; GPU preferred; route=held-out retrieval quality evaluation; source_revision=914f7f89142e33e77833254d9c9b90c3cef7303b; batch_size=1; concurrency=1; 未記録: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E6](../validation/embeddinggemma2-int4-int8-20261018.md) | 2026-10-18 / legacy / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 同じPTQの20ベクトル照合。元精度からの品質劣化を保持。iPhoneはロックで実行できなかった。 · variant=INT4 text / INT8 vision PTQ; backend=Core AI; GPU preferred; route=held-out retrieval quality evaluation; source_revision=914f7f89142e33e77833254d9c9b90c3cef7303b; artifact_revision=7c37ea823af7a4028ec3757e124b53fe8e16a75c99c46d24ca0a7007cca55b85; code_revision=71fb61afd24e1fc5324f57a1abc02744d07ae09c; batch_size=1; concurrency=1; 未記録: warmups, power, thermal, plan_sha256 |
| [E7](../validation/ios-models-integration-20261018.md#acquisition) | 2026-10-19 / legacy / diagnostic | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | ライブラリの更新で、既存のDocumentsの2フォルダから両変換版（定義の13・16ファイル、913,677,333 B・341,880,041 B）を大きさとSHA-256の照合後に移行し、検査した。モデルは読み込まない。 · variant=safe FP16 S128 / vision70; INT4/INT8 PTQ; route=Models self-test (--acquire all); source_revision=914f7f89142e33e77833254d9c9b90c3cef7303b; artifact_revision=daab6dbfbae1610a87d9f181458419e3ea5ce682; 未記録: backend, code_revision, batch_size, concurrency, warmups, power, thermal, plan_sha256 |

### 導入・制限・進捗

- [検索ガイド（英語）](../embeddinggemma2.md)

- [#210](https://github.com/oka-yuji/coreai-samples/issues/210)
<!-- model-report:end -->
