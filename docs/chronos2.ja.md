# Chronos-2

[ホーム](../README.ja.md) · [English](chronos2.md)

Amazon Chronos-2を、単系列・16点先の分位予測としてCore AIのFP32へ変換する再現手順です。MacのPython Core AI経路は固定した合成18条件に合格しています。Swift runtime、Models、NYCサンプルの追加は[#548](https://github.com/oka-yuji/coreai-samples/issues/548)で進めます。

## 変換の契約

[保守用レシピ](../Tools/Conversion/recipes/chronos2/README.md)へ固定した公式snapshotを渡します。観測は1〜8192点、欠損と左paddingはNaN。native入力は16の倍数長のFloat32[1,T]、出力はFloat32[1,21,16]、stateなしです。このprofileは共変量、cross-learning、量子化、16点を超える予測を公開しません。

このリポジトリで重みは配布しません。レシピはローカルbundleとinventoryを作ります。新規変換では資産やmanifestのhashが変わり得るため、そのbundleを検証してから信頼するcatalogを更新します。

## 変換の検証

[検証条件と失敗記録](validation/chronos2-conversion-20261111/README.md)で、変換の一致と用途の品質を区別しています。nativeの6,048出力すべてが `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)` を満たしました。タクシー需要の精度、性能、iPhoneアプリの動作を示す結果ではありません。

## Swift runtime

`ChronosCore`が系列と全21分位を検査し、`ChronosRuntime`が同時に1要求だけを所有します。停止はload/推論の終了とモデル破棄を待ちます。[MacのSwift検証](validation/chronos2-swift-20261112/README.md)は同じ18条件と解放/再ロードに合格しました。iOS libraryのビルドは成功していますが、iPhone推論と通常デモは未確認です。

```sh
swift run --package-path CoreAISamplesKit chronos-verify --help
```

モデル実行時は`AGENTS.md`の共有ロックを使います。JSON入力は`{"context":[1,2,null,4]}`で、`null`が欠損です。結果は補正前の全`[1,21,16]`値とidentityを保持します。出力は既存フォルダ内の新しいファイルに限ります。

## Modelsへの取り込みとプレビュー

Mac/iOSのModelsにChronos-2とNYC固定タクシー乗車件数予測を表示します。**Import model**から、準備済みのChronos-2 FP32フォルダ全体を選びます。9ファイルで478,864,956バイト（約479 MB）あり、管理コピーの分の空き容量も必要です。このビルドにダウンロード先はありません。Modelsの起動・取り込み・検査で推論は始まりません。モデルがなくても事前計算プレビューを操作でき、準備後は端末内予測を明示的に実行できます。端末内予測の失敗時に方式を自動変更しません。

固定資産は`chronos2-fp32`、契約は`chronos2-fp32-v1`、原典は`amazon/chronos-2`のrevision `29ec3766d36d6f73f0696f85560a422f50e8498c`です。`chronos2.json`のSHA-256は`786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572`で、元のmanifestの全bytesと9ファイルすべての容量・hashを照合します。この完全一致で、stateなし・Float32 `[1,T]`から`[1,21,16]`へのABIを含むmetadataの契約を固定し、取込時にモデルロードやruntimeのsnapshot作成を行いません。manifestの整形だけでも新しい信頼済みidentityが必要です。

既存のModels storeがコピー・検査してから登録します。改変・欠損・余分なファイル・symlinkを拒否し、置換に失敗しても旧登録と入力フォルダを保持します。利用のleaseがある間は置換・削除を拒否し、最後のleaseを解放してから管理コピーを削除できます。元フォルダは削除しません。

CPU検査はcatalog・preview・履歴窓・停止とownerの境界・合成ファイルによる保存境界を対象にします。実bundleの明示検査は、共有heavy-jobロック下で`COREAI_CHRONOS2_IMPORT_BUNDLE=/path/to/chronos2-bundle swift test --package-path CoreAISamplesKit --filter ChronosImportTests.knownBundle`を使います。推論せず、取り込み・全量検査・lease・不正置換の拒否・元hashの不変を確かめます。既定のskipは取込成功ではありません。実bundle取り込み、アプリbuild、iPhone推論、通常UIの受入は別の証拠が必要です。

## NYC固定デモ

2025年1月の3場面からゾーンと30分の予測区間を選びます。「30分進める」は経過した1区間の実績を公開し、既存予測の基準時刻を保持します。再予測を明示したときだけ、経過済み実績を追加した最新1,008点を入力します。予測開始はstep 0〜12、表示4点、既知の実績16点です。リセットで予測を消します。表示言語・ゾーン・表示区間の変更は推論や入力値の変更を起こしません。

地図・数値・グラフは共通のFeature状態を参照します。モデル入力は21日間、グラフは直近4時間と2時間先です。件数は記録された乗車回数で、乗客数や未充足需要ではありません。10〜90%分位は校正済みの的中率を保証しません。Appleの背景地図には通信が必要ですが、固定データと端末内予測はオフラインで使えます。出典・固定hash・集計・帰属・再現方法は[Tools/NYC](../Tools/NYC/README.md)を参照してください。

端末内予測は全21×16値と検証済みモデルのidentityを保持し、負数の0補正は表示境界だけで行います。事前計算は提供された3×4値として明示し、端末での処理時間は付けません。native loaderは検証コピーのmanifestをModelsの期待identityと照合してからCore AIモデルを構築します。

FeatureStoreがTaskを所有し、Runnerが20系列全体の終了を待ちます。sessionは待機前に許可を失効し、runtime解放まで待ってからModelsへ停止を返します。sessionごとに新しいServiceを使い、previewもStoreで寿命を管理します。

D5の明示試験は共有heavy-jobロック下で、`COREAI_CHRONOS2_DEMO_BUNDLE`に実bundle、`COREAI_CHRONOS2_DEMO_REFERENCE`に固定公式参照を渡し、`swift test --package-path CoreAISamplesKit --filter ChronosDemoNativeTests.fixedScenes`を実行します。任意の`COREAI_CHRONOS2_DEMO_DATA`は固定NYCData、`COREAI_CHRONOS2_DEMO_REPORT`は新しい結果ファイルです。省略時は同梱データを使い、結果ファイルは上書きしません。60系列・20,160値を固定atol/rtol各1e-4で照合し、stop後のlease 0を要求します。

workerでは実モデルを動かしていません。親担当が原データ再現、必須検査、完全なproject生成、Mac Release、署名iOS、実モデル照合、通常UIと実機の本人受入を行います。既定skipやCPU成功はD5/D7の合格ではありません。下の生成済み評価表は、追加の証拠が登録されるまで従来の結果を保持します。

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](benchmarking.ja.md) · [元の記録](model-reports/records/chronos2.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | — | Modelsでの導入と明示的なプレビューを実装。予測画面はまだ未接続。 | — |
| ios | — | Modelsでの導入と明示的なプレビューを実装。予測画面はまだ未接続。 | — |

### 性能と品質

| 指標 | 結果 | 集計・件数 | 条件と根拠 |
|---|---|---|---|
| バンドル・資産容量 | 478,864,956 B | single; n=1 | [E1](validation/chronos2-conversion-20261111/README.md) · 参照ベクトルとmetadataを含む9ファイルのローカルbundle。実行メモリではありません。 |
| 特殊化キャッシュ容量 | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 初回ロード（特殊化を含む） | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 特殊化済みキャッシュからのロード | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 要求・パイプライン完了時間 | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| ピークRSS | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| ピークphysical footprint | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 推論中の常駐メモリ | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 解放後メモリの基準との差 | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 課題達成率 | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 参照との誤差（定義は根拠参照） | 0.00152587891 error | max; n=6048 | [E1](validation/chronos2-conversion-20261111/README.md) · 全scalarの最大絶対差。各値はatol 1e-4 + rtol 1e-4を満たす。nは比較したscalar数で独立run数ではありません。 |
| 停止受付から完了までの時間 | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 予測のMAE | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 予測のWAPE | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 合格（記載範囲） | 公式FP32、Python Core AI、Swift Core AIが同じ18条件に合格。 | [E1](validation/chronos2-conversion-20261111/README.md) · [E2](validation/chronos2-swift-20261112/README.md) |
| 用途の品質 | 未計測 | 未使用データの予測品質と通常アプリの動作は、数値一致では確認していません。 | — |
| 連続利用・資源解放 | 合格（記載範囲） | Mac Swiftの18要求と1回の解放/再ロードに合格。所有と遅い完了はCPU fakeで検査。メモリ計測・通常アプリ/iPhone受入は未実施。 | [E2](validation/chronos2-swift-20261112/README.md) |
| 停止・再試行・回復 | 未計測 | 取消と停止joinはCPU fakeで検査済み。実モデルの取消と通常アプリは未計測。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](validation/chronos2-conversion-20261111/README.md) | 2026-11-11 / standard / conversion | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 Build version 27B52; Swift 6.4.2; macOS SDK 27.2; Python 3.11.9; chronos-forecasting 2.3.2; torch 2.9.0; coreai-torch 0.4.3; coreai-core 1.0.0b3 | 固定の合成18系列、raw長1〜8192、欠損・定数・周期・傾向。21分位×16点を全件比較。条件ごとに1回、warmupなし。数値一致だけの検査。 · variant=coreai-fp32-univariate; backend=Python Core AI GPU preference; actual delegate not profiled; route=Chronos-2 recipe: independent official CPU reference -> Core AI native verification -> bundle; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=e749b5653298b826770f57ac6a814967fad7162a; batch_size=1; concurrency=1; warmups=0; power=AC Power sampled before and after the run; thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; same after the run; temperature and GPU utilization not measured |
| [E2](validation/chronos2-swift-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; macOS/iOS SDK 27.2; Debug package | 固定18条件・6048原値、その後の解放/再ロードで先頭336値を再照合。warmupなしの数値検証で性能比較ではない。電源/熱とbenchmark計画hashは未記録。 · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=chronos-verify --verify; separate opt-in native test; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=25903277e60e99bfd85e217a96d69b18df82852c; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal, plan_sha256 |
| [E3](validation/chronos2-models-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package tests; Release Mac/iOS applications | 固定9ファイルの導入、5種類の不正置換拒否、leaseと元hash保持。通常Mac画面で導入・プレビュー、署名iOS build。推論・性能計測は行わない。 · variant=coreai-fp32-univariate; backend=ManagedModelStore and normal Mac Models UI; no inference; route=Local bundle import -> inspect -> ready -> lease/release; explicit preview; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=d68828d601fea9d8b6756cc7fe2b2200749d7fa4; 未記録: batch_size, concurrency, warmups, power, thermal, plan_sha256 |

### 導入・制限・進捗

- [変換レシピ](../Tools/Conversion/recipes/chronos2/README.md)

- [変換の検証記録](validation/chronos2-conversion-20261111/README.md)

- [Swift実行の検証記録](validation/chronos2-swift-20261112/README.md)

- [Models導入の検証記録](validation/chronos2-models-20261112/README.md)

- [実装計画 #548](https://github.com/oka-yuji/coreai-samples/issues/548)
<!-- model-report:end -->
