# Chronos-2

[ホーム](../README.ja.md) · [English](chronos2.md)

Amazon Chronos-2を、単系列・16点先の分位予測としてCore AIのFP32へ変換した公開版です。Modelsから[配布bundle](https://huggingface.co/okayuji/chronos-2-coreai-fp32)を取得してNYCタクシーデモへ渡せます。Macの新規取得と本番serviceのnative予測を検証済みで、通常の取得画面・実iPhone・本人受入は未確認です。[公開検証記録](validation/chronos2-download-20261113/README.md)に範囲と失敗経緯を残しています。

## 変換の契約

[保守用レシピ](../Tools/Conversion/recipes/chronos2/README.md)へ固定した公式snapshotを渡します。観測は1〜8192点、欠損と左paddingはNaN。native入力は16の倍数長のFloat32[1,T]、出力はFloat32[1,21,16]、stateなしです。このprofileは共変量、cross-learning、量子化、16点を超える予測を公開しません。

重みは[Hugging Face](https://huggingface.co/okayuji/chronos-2-coreai-fp32)で別配布します。レシピはローカルbundleとinventoryを作ります。新規変換では資産やmanifestのhashが変わり得るため、そのbundleを検証してから信頼するcatalogを更新します。

## 変換の検証

[検証条件と失敗記録](validation/chronos2-conversion-20261111/README.md)で、変換の一致と用途の品質を区別しています。nativeの6,048出力すべてが `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)` を満たしました。タクシー需要の精度、性能、iPhoneアプリの動作を示す結果ではありません。

## Swift runtime

`ChronosCore`が系列と全21分位を検査し、`ChronosRuntime`が同時に1要求だけを所有します。停止はload/推論の終了とモデル破棄を待ちます。[MacのSwift検証](validation/chronos2-swift-20261112/README.md)は同じ18条件と解放/再ロードに合格しました。固定NYCの本番service照合と通常Mac画面も確認済みです。実iPhoneの推論は未確認です。

```sh
swift run --package-path CoreAISamplesKit chronos-verify --help
```

モデル実行時は`AGENTS.md`の共有ロックを使います。JSON入力は`{"context":[1,2,null,4]}`で、`null`が欠損です。結果は補正前の全`[1,21,16]`値とidentityを保持します。出力は既存フォルダ内の新しいファイルに限ります。

## Modelsからの取得・取り込みとプレビュー

**Models**で**Chronos-2**を選び、**モデルをダウンロード**を押します。固定した9ファイル・478,864,956バイト（約479 MB）を匿名で取得し、すべてを検査します。管理用・実行用コピーの空き容量も必要です。既知のChronos-2 FP32フォルダ全体を選ぶ**モデルを取り込む**も利用できます。既存の正常な取り込み済み資産は、そのまま使えます。

**モデル準備完了**になったらNYCタクシーデモを開き、**端末内予測**を選んで**この端末で予測**を押します。Modelsの起動・取得・取り込み・検査・準備で予測を自動実行しません。モデルがなくても**事前計算プレビュー**を操作できます。端末内予測が失敗しても方式を自動変更しません。Mac/iOSに同じ取得経路を接続し、署名iOS buildも確認済みですが、実iPhoneの取得と推論は未確認です。

アプリはHF commit `d3ead488046498faa8b3b5103b1923e52a313d0e`（`v1.0.0`）の`bundle/`を取得します。公開版`v1.0.1`は`f422aa3a7c1173566c99fbfceb97e874b0121dab`で、変更はモデルカードだけです。9ファイルのbundleは全bytesが同一で、両方の公開版とアプリの固定先を保持しています。

固定資産は`chronos2-fp32`、契約は`chronos2-fp32-v1`、原典は`amazon/chronos-2`のrevision `29ec3766d36d6f73f0696f85560a422f50e8498c`です。`chronos2.json`のSHA-256は`786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572`で、元のmanifestの全bytesと9ファイルすべての容量・hashを照合します。この完全一致で、stateなし・Float32 `[1,T]`から`[1,21,16]`へのABIを含むmetadataの契約を固定し、取込時にモデルロードやruntimeのsnapshot作成を行いません。manifestの整形だけでも新しい信頼済みidentityが必要です。

40桁のHF commitは取得するbytesを選び、64桁のmanifest hashは信頼する資産・保存先・leaseを識別します。両方の契約を検査し、取得先を追加しても`localManifest`のidentityと取り込み済み登録は変わりません。HTTPの取得revisionにmanifest hashを使うことはありません。

既存のModels storeがコピー・検査してから登録します。改変・欠損・余分なファイル・symlinkを拒否し、置換に失敗しても旧登録と入力フォルダを保持します。利用のleaseがある間は置換・削除を拒否し、最後のleaseを解放してから管理コピーを削除できます。元フォルダは削除しません。

CPU検査はcatalog・preview・履歴窓・停止とownerの境界・合成ファイルによる保存境界を対象にします。実bundleの明示検査は、共有heavy-jobロック下で`COREAI_CHRONOS2_IMPORT_BUNDLE=/path/to/chronos2-bundle swift test --package-path CoreAISamplesKit --filter ChronosImportTests.knownBundle`を使います。推論せず、取り込み・全量検査・lease・不正置換の拒否・元hashの不変を確かめます。既定のskipは取込成功ではありません。実bundle取り込み、アプリbuild、iPhone推論、通常UIの受入は別の証拠が必要です。

[公開DLの検査](validation/chronos2-download-20261113/README.md)は空のmanaged store・移行候補なしから実取得し、9ファイル全件を照合してから本番Models/Store/session/serviceで固定60場面・ゾーンの20,160原値を比較しました。元hash不変とclose後lease0も確認しています。同じ固定参照による変換一致の検査で、独立した予測品質や性能の評価ではありません。共有ロック下で新規`COREAI_CHRONOS2_DOWNLOAD_OUTPUT`、hash検査用の元bundleを`COREAI_CHRONOS2_DOWNLOAD_SOURCE`、固定参照を`COREAI_CHRONOS2_DOWNLOAD_REFERENCE`へ渡し、`swift test --package-path CoreAISamplesKit --filter ChronosDownloadNativeTests.freshModelsDownload`で再現します。既定はskipで、通常GUI・実iPhone・本人受入は別に確認します。

## NYC固定デモ

2025年1月の3場面からゾーンと30分の予測区間を選びます。「30分進める」は経過した1区間の実績を公開し、既存予測の基準時刻を保持します。再予測を明示したときだけ、経過済み実績を追加した最新1,008点を入力します。予測開始はstep 0〜12、表示4点、既知の実績16点です。リセットで予測を消します。表示言語・ゾーン・表示区間の変更は推論や入力値の変更を起こしません。

地図・数値・グラフは共通のFeature状態を参照します。モデル入力は21日間、グラフは直近4時間と2時間先です。件数は記録された乗車回数で、乗客数や未充足需要ではありません。10〜90%分位は校正済みの的中率を保証しません。Appleの背景地図には通信が必要ですが、固定データと端末内予測はオフラインで使えます。出典・固定hash・集計・帰属・再現方法は[Tools/NYC](../Tools/NYC/README.md)を参照してください。

端末内予測は全21×16値と検証済みモデルのidentityを保持し、負数の0補正は表示境界だけで行います。事前計算は提供された3×4値として明示し、端末での処理時間は付けません。native loaderは検証コピーのmanifestをModelsの期待identityと照合してからCore AIモデルを構築します。

FeatureStoreがTaskを所有し、Runnerが20系列全体の終了を待ちます。sessionは待機前に許可を失効し、runtime解放まで待ってからModelsへ停止を返します。sessionごとに新しいServiceを使い、previewもStoreで寿命を管理します。

D5の明示試験は共有heavy-jobロック下で、`COREAI_CHRONOS2_DEMO_BUNDLE`に実bundle、`COREAI_CHRONOS2_DEMO_REFERENCE`に固定公式参照を渡し、`swift test --package-path CoreAISamplesKit --filter ChronosDemoNativeTests.fixedScenes`を実行します。任意の`COREAI_CHRONOS2_DEMO_DATA`は固定NYCData、`COREAI_CHRONOS2_DEMO_REPORT`は新しい結果ファイルです。省略時は同梱データを使い、結果ファイルは上書きしません。60系列・20,160値を固定atol/rtol各1e-4で照合し、stop後のlease 0を要求します。

[検証記録](validation/chronos2-demo-20261112/README.md)に原データ再現、実モデル照合、通常Mac画面、翻訳とホストbuildの結果を残しています。実iPhoneの推論と本人受入は未確認です。既定skipやCPU成功を実モデル・本人受入の合格には数えません。

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](benchmarking.ja.md) · [元の記録](model-reports/records/chronos2.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | ✅ | 通常MacでNYC固定/API/履歴と商品売上予測を確認。公開版の新規取得と本番serviceのnative照合は別途成功。通常DL画面と本人受入は未確認。 | [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) · [E6](validation/chronos2-journal-20261112/README.md) · [E7](validation/chronos2-sales-20261113/README.md) · [E8](validation/chronos2-download-20261113/README.md) · [E9](validation/chronos2-download-20261113/README.md) |
| ios | △ | NYC/商品売上の画面・native adapterとDL/取込を接続し、署名build・同梱取得先を検査。実iPhoneの取得・推論と本人受入は未確認。 | [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) · [E6](validation/chronos2-journal-20261112/README.md) · [E7](validation/chronos2-sales-20261113/README.md) · [E8](validation/chronos2-download-20261113/README.md) |

### 性能と品質

| 指標 | 結果 | 集計・件数 | 条件と根拠 |
|---|---|---|---|
| バンドル・資産容量 | 478,864,956 B | single; n=1 | [E1](validation/chronos2-conversion-20261111/README.md) · 参照ベクトルとmetadataを含む9ファイルのローカルbundle。実行メモリではありません。 |
| 特殊化キャッシュ容量 | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 初回ロード（特殊化を含む） | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 特殊化済みキャッシュからのロード | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 要求・パイプライン完了時間 | 未計測 | — | 制御した時間benchmarkは未計測。新sessionの検証/ロードを含む売上の機能確認中の要求時間は売上根拠へ別記。 |
| ピークRSS | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| ピークphysical footprint | 212,411,176 B | single; n=1 | [E7](validation/chronos2-sales-20261113/README.md) · 取込と売上2回分を含むDebug試験process一つのkernel生涯peak。要求単位・モデル単独・GPUメモリ・iPhone予算ではない。 |
| 推論中の常駐メモリ | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 解放後メモリの基準との差 | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 課題達成率 | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 参照との誤差（定義は根拠参照） | 0.00152587891 error | max; n=6048 | [E1](validation/chronos2-conversion-20261111/README.md) · 全scalarの最大絶対差。各値はatol 1e-4 + rtol 1e-4を満たす。nは比較したscalar数で独立run数ではありません。 |
| 参照との誤差（定義は根拠参照） | 8.39233398e-05 error | max; n=10752 | [E7](validation/chronos2-sales-20261113/README.md) · 売上の固定16窓を再load後も照合。全21×16原値が固定atol/rtol各1e-4以内。反復scalar数で、独立した品質標本数ではない。 |
| 参照との誤差（定義は根拠参照） | 0.000228881836 error | max; n=20160 | [E8](validation/chronos2-download-20261113/README.md) · 新規取得bundleで既存の固定NYC60条件を照合。原値の最大絶対差で、全値がatol 1e-4 + rtol 1e-4内。独立した用途品質や反復run数ではありません。 |
| 停止受付から完了までの時間 | 未計測 | — | この変換一致検査では未計測。アプリ動作・用途品質・性能は別の証拠で確認します。 |
| 予測のMAE | 4.44884534 count | aggregate; n=212 | [E7](validation/chronos2-sales-20261113/README.md) · 固定売上の初回のみ、共通有効212/224ペア。負値補正後・表示丸め前の中央値。季節ナイーブは根拠参照。独立した品質評価ではない。 |
| 予測のWAPE | 0.901678022 ratio | aggregate; n=212 | [E7](validation/chronos2-sales-20261113/README.md) · 同じ固定売上ペアと共通分母。季節ナイーブ比較、ゼロ分母/欠損の扱いは根拠参照。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 合格（記載範囲） | 合成18条件・固定NYC60場面/ゾーン・実API20系列が凍結した原tensor照合に合格。API入力は固定金曜の条件と重複。商品売上は固定16窓を解放/再表示後も公式FP32と照合。 公開版の新規取得でも同じNYC60条件を再照合し、未使用系列の追加とはしない。 | [E1](validation/chronos2-conversion-20261111/README.md) · [E2](validation/chronos2-swift-20261112/README.md) · [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) · [E7](validation/chronos2-sales-20261113/README.md) · [E8](validation/chronos2-download-20261113/README.md) · [E9](validation/chronos2-download-20261113/README.md) |
| 用途の品質 | 未計測 | 固定売上の診断指標は記録したが、数値一致や過去デモ窓を未使用データの予測品質とは扱わない。 | — |
| 連続利用・資源解放 | 合格（記載範囲） | NYCと売上で解放/再load、本番serviceのlease0、通常Macの離脱/再表示を確認。売上では点のメモリとprocess生涯peakを記録。iPhone受入は未確認。 公開版の新規取得/native予測も元hash不変とlease解放を確認。 | [E2](validation/chronos2-swift-20261112/README.md) · [E4](validation/chronos2-demo-20261112/README.md) · [E5](validation/chronos2-api-20261112/README.md) · [E7](validation/chronos2-sales-20261113/README.md) · [E8](validation/chronos2-download-20261113/README.md) · [E9](validation/chronos2-download-20261113/README.md) |
| 停止・再試行・回復 | 未計測 | 取消と停止joinはCPU fakeで検査済み。実モデルの取消と通常アプリは未計測。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](validation/chronos2-conversion-20261111/README.md) | 2026-11-11 / standard / conversion | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 Build version 27B52; Swift 6.4.2; macOS SDK 27.2; Python 3.11.9; chronos-forecasting 2.3.2; torch 2.9.0; coreai-torch 0.4.3; coreai-core 1.0.0b3 | 固定の合成18系列、raw長1〜8192、欠損・定数・周期・傾向。21分位×16点を全件比較。条件ごとに1回、warmupなし。数値一致だけの検査。 · variant=coreai-fp32-univariate; backend=Python Core AI GPU preference; actual delegate not profiled; route=Chronos-2 recipe: independent official CPU reference -> Core AI native verification -> bundle; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=e749b5653298b826770f57ac6a814967fad7162a; batch_size=1; concurrency=1; warmups=0; power=AC Power sampled before and after the run; thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; same after the run; temperature and GPU utilization not measured |
| [E2](validation/chronos2-swift-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; macOS/iOS SDK 27.2; Debug package | 固定18条件・6048原値、その後の解放/再ロードで先頭336値を再照合。warmupなしの数値検証で性能比較ではない。電源/熱とbenchmark計画hashは未記録。 · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=chronos-verify --verify; separate opt-in native test; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=25903277e60e99bfd85e217a96d69b18df82852c; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal, plan_sha256 |
| [E3](validation/chronos2-models-20261112/README.md) | 2026-11-12 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package tests; Release Mac/iOS applications | 固定9ファイルの導入、5種類の不正置換拒否、leaseと元hash保持。通常Mac画面で導入・プレビュー、署名iOS build。推論・性能計測は行わない。 · variant=coreai-fp32-univariate; backend=ManagedModelStore and normal Mac Models UI; no inference; route=Local bundle import -> inspect -> ready -> lease/release; explicit preview; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=d68828d601fea9d8b6756cc7fe2b2200749d7fa4; 未記録: batch_size, concurrency, warmups, power, thermal, plan_sha256 |
| [E4](validation/chronos2-demo-20261112/README.md) | 2026-11-12 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac application | 固定60場面/ゾーン・20160原値。通常Mac画面で端末内予測・比較・離脱を確認。停止とowner境界はCPU fakeでも検査。iPhone推論・本人受入・性能計測は未実施。電源/熱とbenchmark計画hashは未記録。 · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Normal Mac Models -> explicit on-device prediction -> interval and clock changes -> leave/reopen; separate production-service native comparison; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=06e010133afd10d0d31532cfa804b97818944b30; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal, plan_sha256 |
| [E5](validation/chronos2-api-20261112/README.md) | 2026-11-12 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac application | 本番clientで1月全29,760値を照合。native経路は金曜20,160入力・6,720原値とclose後lease0を確認。通常Macで明示取得/予測/実績と月末停止を確認。チャートAX修正の版は別記。制御した性能計測とiPhone推論は未実施。 · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Production Models/Store/Service with a pinned January API reference; separate normal Mac GUI; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=02aa27facf98b6d7d503f72a06e8ac0b123dec01; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal, plan_sha256 |
| [E6](validation/chronos2-journal-20261112/README.md) | 2026-11-12 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac application | 固定金曜1起点の20,160入力・6,720原値・80比較ペア。保存ファイルから独立計算とhash確認、lease0。通常Macの保存/再起動/実績確認とlocale修正は版を別記。独立した精度評価・制御した性能計測ではなく、電源/熱とbenchmark計画hashは未記録。 · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Production Models/Store/service/session -> live verified API -> native prediction -> immutable journal -> separate actual check -> reload -> stop; separate normal Mac GUI; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=48adfc21c80ff7ceb1828f5b7326fde1c760d0be; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal, plan_sha256 |
| [E7](validation/chronos2-sales-20261113/README.md) | 2026-11-13 / legacy / native_app | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package and Release Mac/iOS builds | 4商品×4過去起点、365点履歴、2回/10752原値照合。初回の比較212/224ペア。通常Macの軸修正は1954168。時間とprocess全体メモリは診断値で、制御したbenchmarkや未使用データの品質評価ではない。 · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Production Models/Store/Service/Sessions with original sales CSV; independent raw/metric audit; separate normal Mac sidebar; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=55df3378a40a98f8e34c6fafa8e8a33bb1001e71; batch_size=1; concurrency=1; warmups=0; power=AC Power before and after; thermal=pmset reported no thermal/performance warnings before and after; temperature and GPU load unmeasured; 未記録: plan_sha256 |
| [E8](validation/chronos2-download-20261113/README.md) | 2026-11-13 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package tests; Release Mac/iOS build checks | d3ead488から匿名9GET・478864956バイトを新規取得。固定60場面/ゾーン・20160原値、元hash不変とlease0を照合。同じ固定参照の再利用で、新しい独立評価データではない。ホスト識別は実行後に取得、電源/熱・benchmark計画hashは未記録。GUI・iPhone・本人受入は別。 · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Fresh public Models download -> full validation -> production Store/sessions/service -> fixed native forecasts -> close; drawing-only adapter; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=9c7da3d0de00ef86f8ca1235385a254bd82f7982; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal, plan_sha256 |
| [E9](validation/chronos2-download-20261113/README.md) | 2026-11-13 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Swift 6.4.2; SDK 27.2; Debug package tests; Release Mac/iOS build checks | 売上統合とopt-in catalog修正後の再取得。既存と同じ60場面/ゾーン・20160原値で、未使用データの追加ではない。全case集合・Float32復元原値・元hash不変・lease0を確認。Test50.129秒とwrapper63.754719333秒は検査全体で性能値ではない。 · variant=coreai-fp32-univariate; backend=Swift Core AI GPU preference; actual delegate not profiled; route=Fresh production Models download -> corrected production-linked catalog -> Store/Sessions/Service -> same60 fixed NYC cases -> close; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=ec02f94bde596787ef83c837499732b9c76f3c2a; batch_size=1; concurrency=1; warmups=0; 未記録: power, thermal, plan_sha256 |

### 導入・制限・進捗

- [変換レシピ](../Tools/Conversion/recipes/chronos2/README.md)

- [変換の検証記録](validation/chronos2-conversion-20261111/README.md)

- [Swift実行の検証記録](validation/chronos2-swift-20261112/README.md)

- [Models導入の検証記録](validation/chronos2-models-20261112/README.md)

- [固定NYCデモの検証記録](validation/chronos2-demo-20261112/README.md)

- [公式APIデモの検証記録](validation/chronos2-api-20261112/README.md)

- [実装計画 #548](https://github.com/oka-yuji/coreai-samples/issues/548)

- [予測履歴の検証記録](validation/chronos2-journal-20261112/README.md)

- [TukuruSFの実行分析](validation/chronos2-tsf-20261112/README.md)

- [商品売上デモ](chronos2-sales.md)

- [商品売上の検証](validation/chronos2-sales-20261113/README.md)

- [公開ダウンロードの検証記録](validation/chronos2-download-20261113/README.md)
<!-- model-report:end -->

## 検証済みのNYC API日時

同梱3場面に加えて「検証済みのNYC API日時」を選べる。「検証済みの日時を確認」からニューヨークの日付と30分刻みの時刻を選び、21日分の履歴を取得する。API予測にはModelsの準備済み資源と「端末内予測」の明示選択が必要。予測成功後に実績を別途取得する。Modelsを開く操作や取得元の切替だけでは通信・推論を開始しない。

同梱した独立照合の根拠はマンハッタンの20ゾーンと次の期間に限る。カタログ掲載年度の全域を提供するものではない。

| 境界 | ニューヨーク時間 | UTC |
|---|---|---|
| 検証済みデータの開始（含む） | 2025-01-01 00:00 EST | 2025-01-01T05:00:00Z |
| 検証済みデータの終端（含まない） | 2025-02-01 00:00 EST | 2025-02-01T05:00:00Z |
| 最初の予測起点 | 2025-01-22 00:00 EST | 2025-01-22T05:00:00Z |
| 最後の予測起点（含む） | 2025-02-01 00:00 EST | 2025-02-01T05:00:00Z |

予測起点には直前1,008区間が必要なので、検証済みのデータ期間と選択可能な起点は異なる。datasetは`irx2-vgm4`、revisionは`2026-10-01T16:57:47.000Z`に固定する。掲載年度でも未知のrevisionや根拠の範囲外は**未検証**のnull、カタログにない年度は**未公開**のnullとする。どちらも0件にせず、集計を要求しない。21日履歴に含まれれば対応する理由で拒否する。1月末の実績16区間は、検証済みの先頭部分と未検証の末尾を保持し、先頭部分だけ予測を進められる。更新日だけで新しいrevisionを自動承認しない。

APIの年次公開とTLC月次Parquetは別の公開である。年次metadataは投入完了の印ではない。公式domain・owner・provenance・年別名称・公開状態・TLC agency・列型・catalog revisionを引き続き検査し、集計要求の前後に実metadataを取得して、両方が固定revisionと一致することを確認する。これはproviderがrevisionへ反映する変更を検出する契約で、トランザクションとしての一貫性を保証しない。さらに要求した各検証済み区間の件数を独立参照と完全一致で照合する。参照が正数なのに応答が空なら検証エラー。参照が0件の区間にgroupがなければ観測済みの0件。失敗時は前の有効な表示と予測モードを保つ。

### 1月の独立照合と最初の不一致

[参照の正確なbytes](../CoreAISamplesKit/Sources/ChronosDemoRuntime/Resources/NYCVerified/january-2025.json)はruntimeが所有し、既存の固定場面`NYCData`とは別に置く。SHA-256は`2da47735c51b5987dec9f6bee72c97acd39abe4870991e413dca30a1912c6530`、88,976 bytes。dataset ID・revision・半開区間・1,800秒間隔・`America/New_York`・重複しない20ゾーン・原典2ファイルのhashを含む。読込時にbytesの固定hashとサイズ、identity、日時境界、形状、有限な非負整数の件数、ゾーン集合を検査する。純粋なFeatureには検証方針の値だけを渡し、bundle IOを置かない。別期間の追加には、その期間を独立照合した資源と明示的な方針登録が必要。

| 原典 | SHA-256 |
|---|---|
| [2025年1月Parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet) | `9af277e4c0d3f9deb30644da822981e1e7df6af58313170fd3aa8a474485488a` |
| [2025年2月Parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-02.parquet) | `037cba555a73663f3a51a2c27816e40e3feb364769942bdf122b9da31e377bd3` |

親担当の独立集計では、20ゾーン×1月の1,488区間＝29,760値が完全一致した。APIは5回に分けて取得し、前後のrevisionは同一だった。最初の1月ファイル単独の比較は**月末9セルで不合格**。2月ファイルの1月乗車30件のうち対象ゾーンの10件を含めると全差分が解消した。不合格だった参照のhashは`68d9a72188d1fb7bd8a3e6b07990c4163b404a67b233427eedfb15ea4dfdde55`。この結果を年度全域の数値検証へ広げない。

| ゾーン | 1月31日の現地開始時刻 | API | 1月ファイル単独 |
|---|---|---:|---:|
| 186 | 23:00 | 54 | 53 |
| 234 | 23:30 | 163 | 162 |
| 246 | 23:30 | 36 | 35 |
| 48 | 22:30 | 152 | 151 |
| 48 | 23:00 | 113 | 111 |
| 50 | 23:30 | 25 | 24 |
| 68 | 22:00 | 143 | 142 |
| 68 | 23:30 | 91 | 90 |
| 90 | 23:30 | 88 | 87 |

### 欠損、出典、再利用

区間はUTCで1,800秒ずつ進む。春に存在しない現地時刻は選択肢に入れず、その時刻の不正なprovider行は拒否し、有効なUTC区間へ割り当て直さない。親担当が20ゾーンの2025-03-09 02:00〜03:00を照会した結果は0groupだった。他の年度にも不正行がないという証明ではない。春は現在の検証済み範囲外であり、夏時間と不正行のCPU試験には明示的な合成参照を使う。検証済みの範囲内に秋の重複時刻があれば両方とも欠損にする。観測0・未検証・未公開・重複時刻・未取得を区別する。

`NYCAPIClient`が範囲の世代、通信、一時キャッシュを所有する。固定2ホストのHTTPSだけを使い、転送・認証を拒否し、受信途中に8 MiBを超えたら停止する。集計行は50,000未満。検証済み区間との交差だけを、NY年境界とUTCの最大7日で分けて問い合わせる。明示的な一過性エラーだけ1回再試行する。元HTTP応答bytesとそのSHA-256、URL・区間・status・取得時刻・catalog・取得前後のmetadataを保持する。独立参照のhashをHTTP元応答hashへ置き換えない。

キャッシュキーと保存windowには検証方針・参照identityと前処理版`nyc-utc30m-verified-v3`を含める。旧版の年全域キャッシュは新しい方針を迂回できない。読込前の種類・サイズ検査に加え、元bytesのhash、metadata、現在の承認済み方針、再構成した件数と参照の完全一致を再検査する。保存完了後は最大32件・合計64 MiB、1件の元bytesは32 MiB、符号化済みファイルは48 MiBまで。孤立したUUID一時ファイルの回収・保存・pruneを同じ排他lockで守り、保存中は一時的に符号化済みファイル1件分が追加される。

API予測はゾーンごとのnative全21分位×16点と不変の入力snapshotを保持する。決定的なsorted-key JSONのSHA-256にはUTC開始・起点、nullのある件数、revision、前処理、検証identity、取得証拠が含まれる。snapshotも同じ方針と元応答で再検証する。後の入力には明示取得した実績のうち経過済みで検証済みの区間だけを、既存の重複時刻マスクごと加える。実績の追加取得で以前のsnapshotを変えない。キャッシュは永続的な予測記録ではない。

### 検証

範囲境界、破損、未知revision、正数参照への空応答、観測0、前後metadata、取消、キャッシュのCPU検査と提供応答の再生を実行する。

```sh
swift test --package-path CoreAISamplesKit --filter 'ChronosVerifiedRangeTests|ChronosAPI|ChronosFeature'
COREAI_CHRONOS2_API_FIXTURES=/path/to/fixtures swift test --package-path CoreAISamplesKit --filter ChronosAPIReplayTests
```

fixtureは`api-baseline`・`api-fixed-history`・`api-fixed-actuals`を含む。履歴20,160値と実績80値を本番の検証参照で照合し、元HTTP hashも維持する。実績fixtureは2時間分で、追加の時間を観測済みと推定しない。9年度metadataの固定再生には`api-baseline`・`api-annual-metadata`を含む記録を`COREAI_CHRONOS2_API_ANNUAL_METADATA`で指定し、`ChronosAPIReplayTests.annualMetadata`を使う。9年度のmetadata合格は9年度全域の数値合格を意味しない。

親担当の実API・native検査は共有heavy-jobロック下で行う。

```sh
COREAI_CHRONOS2_API_BUNDLE=/path/to/chronos2-bundle COREAI_CHRONOS2_API_REFERENCE=/path/to/fixed-scenes-reference.json COREAI_CHRONOS2_API_REPORT=/new/private/report.json swift test --package-path CoreAISamplesKit --filter ChronosAPINativeTests.liveHistoryAndNative
```

2025-01-24 17:00 NY、履歴20,160値、native全6,720値、固定atol/rtol各1e-4、元モデルidentity、close後lease 0の条件を維持する。推論前に`report.history.json`、成功時に`report.json`を新規private出力へ保存する。既定ではnativeをskipする。親担当が29,760値の実取得再照合、通常GUI、署名build、レビューも実施する。CPU再生を実API・native・Mac/iPhoneの通常UI・本人受入へ広げない。上の生成済み表は元の証拠を保持し、親担当が別途更新する。

## 予測の保存と実績照合

端末内で生成した予測は自動保存する。事前計算の表示は新しい記録を作らない。「この端末に予測を保存しました」の表示を確認し、「予測履歴」から記録を選ぶ。アプリを再起動しても記録は残る。「公開済みの実績を確認」は、保存した起点から2時間・全20ゾーンを別操作で取得し、モデルを再実行せずに照合する。原予測と過去のcheckは変更しない。

詳細にはNYの起点、取得元、端末の保存時刻、モデル版、照合状態とペア数を表示する。MAE/WAPEは非負にした丸め前の中央値と、比較可能なペアだけで計算する。実績合計0ではWAPEを出さない。未知revisionや確認済み期間の外は、未チェック・未公開・観測0・秋の時刻比較不能と区別して「独立照合されていない」状態にする。通信失敗を欠損や0件へ変えない。APIの確認済み範囲は2025年1月のまま。

保存失敗時も予測表示を保ち、明示的に再試行できる。再試行は同じUUID・日時・bytesを使う。画面を閉じる・選択を替える・背景へ移る操作は未確定の処理を失効させ、保存済み記録は保持する。検証に通らない記録は読めない状態として表示する。自動削除や別の試作アプリの記録移行は行わない。保存時刻は端末時計によるもので、第三者の時刻証明ではない。

記録はアプリのApplication Support内の`coreai-samples/ChronosJournal-v1`に置く。全20系列の1008入力・21×16原予測、実際に検証した資産identity、元bytes/hash、前処理と取得根拠を保持する。表示と指標だけで負の中央値を非負にする。Journal envelope/2はDateのreference-clock精度を保ち、既存APIのcanonical bytesとは分ける。読込時は外側のhashだけでなく記録全体を検査する。payloadは192MiB、envelopeは256MiB+4KiBまで。ディレクトリ・file・lock初期化でsymlink等の拒否と排他的な公開を保つ。

実結合は共有heavy-jobロックの下で次を実行する。Models/Store/Service/Sessionsと本番の保存処理を通し、固定実APIの20,160入力・6,720原値・80実績、再読込み、原file不変、停止後lease0を確認する。描画だけは検査用adapterで、通常UIや本人受入の代用にはしない。出力先は未作成のフォルダを指定し、通常suiteではskipする。

```sh
COREAI_CHRONOS2_JOURNAL_BUNDLE=/path/to/chronos2-bundle \
COREAI_CHRONOS2_JOURNAL_REFERENCE=/path/to/fixed-scenes-reference.json \
COREAI_CHRONOS2_JOURNAL_OUTPUT=/new/output \
swift test --package-path CoreAISamplesKit --filter ChronosAPINativeTests.liveJournal
```

[保存・照合の検証と失敗履歴](validation/chronos2-journal-20261112/README.md)、[TukuruSF実行分析](validation/chronos2-tsf-20261112/README.md)では、自動検査・通常Mac・統合・未完了の本人/iPhone受入を分けて記録する。

## 商品売上予測

[商品売上デモ](chronos2-sales.ja.md)で原典の4商品、7日/14日予測、欠損日の保持、季節ナイーブ比較を利用できます。原典の非商用条件を示し、データは明示操作で取得します。[実モデル・通常Macの検証](validation/chronos2-sales-20261113/README.md)と実iPhone・本人受入は区別しています。
