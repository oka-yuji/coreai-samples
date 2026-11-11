# Chronos-2

[ホーム](../README.ja.md) · [English](chronos2.md)

Amazon Chronos-2を、単系列・16点先の分位予測としてCore AIのFP32へ変換する再現手順です。MacのPython Core AI経路は固定した合成18条件に合格しています。Swift runtime、Models、NYCサンプルの追加は[#548](https://github.com/oka-yuji/coreai-samples/issues/548)で進めます。

## 変換の契約

[保守用レシピ](../Tools/Conversion/recipes/chronos2/README.md)へ固定した公式snapshotを渡します。観測は1〜8192点、欠損と左paddingはNaN。native入力は16の倍数長のFloat32[1,T]、出力はFloat32[1,21,16]、stateなしです。このprofileは共変量、cross-learning、量子化、16点を超える予測を公開しません。

このリポジトリで重みは配布しません。レシピはローカルbundleと、後続のModels登録用inventoryを作ります。新規変換では資産やmanifestのhashが変わり得るため、そのbundleを検証してから信頼するcatalogを更新します。

## 変換の検証

[検証条件と失敗記録](validation/chronos2-conversion-20261111/README.md)で、変換の一致と用途の品質を区別しています。nativeの6,048出力すべてが `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)` を満たしました。タクシー需要の精度、性能、iPhoneアプリの動作を示す結果ではありません。

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](benchmarking.ja.md) · [元の記録](model-reports/records/chronos2.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | — | ネイティブサンプルは未接続です。変換の成功をアプリ対応済みにはしません。 | — |
| ios | — | ネイティブサンプルは未接続です。変換の成功をアプリ対応済みにはしません。 | — |

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
| 変換・参照との一致 | 合格（記載範囲） | 未変更の公式参照とexport wrapper、実Core AIが固定18条件の判定に合格。 | [E1](validation/chronos2-conversion-20261111/README.md) |
| 用途の品質 | 未計測 | Swift/アプリの実経路と未使用の用途評価は、この変換結果に含みません。 | — |
| 連続利用・資源解放 | 未計測 | Swift/アプリの実経路と未使用の用途評価は、この変換結果に含みません。 | — |
| 停止・再試行・回復 | 未計測 | Swift/アプリの実経路と未使用の用途評価は、この変換結果に含みません。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](validation/chronos2-conversion-20261111/README.md) | 2026-11-11 / standard / conversion | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 Build version 27B52; Swift 6.4.2; macOS SDK 27.2; Python 3.11.9; chronos-forecasting 2.3.2; torch 2.9.0; coreai-torch 0.4.3; coreai-core 1.0.0b3 | 固定の合成18系列、raw長1〜8192、欠損・定数・周期・傾向。21分位×16点を全件比較。条件ごとに1回、warmupなし。数値一致だけの検査。 · variant=coreai-fp32-univariate; backend=Python Core AI GPU preference; actual delegate not profiled; route=Chronos-2 recipe: independent official CPU reference -> Core AI native verification -> bundle; source_revision=29ec3766d36d6f73f0696f85560a422f50e8498c; artifact_revision=786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572; code_revision=e749b5653298b826770f57ac6a814967fad7162a; batch_size=1; concurrency=1; warmups=0; power=AC Power sampled before and after the run; thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; same after the run; temperature and GPU utilization not measured |

### 導入・制限・進捗

- [変換レシピ](../Tools/Conversion/recipes/chronos2/README.md)

- [変換の検証記録](validation/chronos2-conversion-20261111/README.md)

- [実装計画 #548](https://github.com/oka-yuji/coreai-samples/issues/548)
<!-- model-report:end -->
