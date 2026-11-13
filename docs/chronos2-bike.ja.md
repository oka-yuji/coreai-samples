# Seoul Bike予測

[English](chronos2-bike.md)

Seoul Bikeデモは既存のChronos-2 FP32資産で、時間別レンタル件数の予測を実績・週周期の季節ナイーブと比較します。商品売上と同じForecastのStore・Runner・Service・Sessionsとモデル所有を再利用し、NYCタクシー・商品売上は別の入口として維持しています。

Seoul BikeはCore AIのCPU専用実行と`expectFrequentReshapes=false`を使います。固定5起点のGPU優先実行で裾の分位に数値不一致があり、CPU専用のprobeでは変更していないatol/rtol各1e-4を満たしました。モデルファイルと入出力契約は変更せず、商品売上とNYCはGPU優先を維持します。この設定だけで一般的な予測精度やiPhone性能を保証しません。

## 操作

1. Macのサイドバーで **Chronos-2 → Seoul Bike予測**、iOSのModelsでSeoul Bikeデモを開きます。既存のModels取得操作から同じChronos-2資産を準備します。
2. **同梱の自転車データを読み込む** を押します。検証済みサンプルは同梱され、通信は不要です。モデルなしのプレビューでも履歴と過去天候を確認できますが、予測はできません。
3. **Asia/Seoul** の日付・整数時と、**1・4・8・16時間** の期間を選びます。履歴と比較実績がそろう日時だけを提示します。期間を延ばしたとき、末尾の起点は最後の有効な時刻へ調整します。
4. **この端末で予測** を押します。表示・モデル準備・データ読込み・選択変更・表示言語の変更だけでは推論しません。停止と終了は進行中の操作とモデル解放を待ち、Modelsがleaseを返します。

## 入力と表示

`seoul-bike` の1系列は、Asia/Seoulの2017-12-01 00:00〜2018-11-30 23:00の8,760時間です。入力は**起点より前の336時間**の件数だけです。季節ナイーブは**168時間前**の対応する時刻を使います。未来の件数・天候・休日・稼働フラグを入力しません。

グラフは直前72時間、nativeの中央値とP10〜P90、比較用に保持した実績、季節ナイーブを示します。全21分位×16点の原出力、日時・実績・基準値・モデルidentityは保持します。負の予測値は表示と指標だけ0にします。MAE・WAPEは実績・中央値・基準値がそろう同じ組で計算し、比較した時間数と候補時間数を表示します。実績合計が0ならWAPEは算出せず、100%を超える値もそのまま表示します。P10〜P90は実績の包含を保証しません。

気温（°C）・降水量（mm）は別グラフで**起点より前の72時間**だけを示し、最後の観測の休日・稼働状態も表示します。過去の実測値であり、将来予報でもモデル入力でもありません。天候による条件付き予測、24時間予測、予測値の再入力による延長は非対応です。295行の非稼働0件は休止中のレンタル実績であり、潜在需要がないことを意味しません。

## 出典と再現

Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5F62R.

ライセンス: [Creative Commons Attribution 4.0 International（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/)。出典に個人の作成者名は列挙されていません。

加工はWindows-1252（CP1252）からUTF-8への文字コード変換だけです。CRLF改行・列名・行・単位・値は維持し、補完や集計はしていません。同梱の `ForecastRuntime/Resources/SeoulBike/NOTICE.txt` に原表記を保持します。[データ準備手順](../Tools/Forecasting/bike-data.md)に原ZIP、hash、再現コマンドを記載しています。

SwiftPMはSeoulBikeフォルダをCSVの加工なしでコピーします。読込みのたびにCSV・manifest・NOTICEの固定hashを検査し、その後 `BikeDataset.parse` が列・毎時の順序・完全な日付・整数件数・有限天候値・分類・非稼働0を検査します。`pastWeather(originIndex:count:)` は有効な自転車データの起点と、直前1〜336行だけを受け付けます。売上に天候を捏造せず、既存の固定取得先・cache・Romeの暦日契約を維持します。

## 検証の境界

CPU試験は同梱データ、不正入力、履歴と期間の境界、天候とkindの分離、明示操作、遅い停止、再表示時の新規sessionを検査します。監督側の独立verifierは別の原典oracleで全8,760時刻と5組の入力・実績・基準値・過去天候を照合します。通常試験でモデルやネットワークは使用しません。

```sh
swift test --package-path CoreAISamplesKit --force-resolved-versions --filter 'Bike|Forecast|Chronos'
python3 -B -I -m unittest discover -s Tools/Forecasting -p test_bike_source.py -v
```

実モデル試験は共有heavy-jobロック下で明示実行します。

```sh
COREAI_FORECAST_BUNDLE=/path/to/chronos2-bundle COREAI_BIKE_REFERENCE=/path/to/bike-reference.json COREAI_BIKE_REPORT=/new/private/bike-report.json swift test --package-path CoreAISamplesKit --force-resolved-versions --filter ForecastNativeTests.bike
```

固定参照は異なる5起点で `series_id=seoul-bike`、`horizon=16`、336入力と21分位×16点を持ちます。本番Models・Store・Service・Sessionsで全入力と全raw値をatol/rtol各1e-4で照合し、close後lease 0、再表示・再実行を確認します。新規・上書き禁止の非公開reportにraw予測・実績・基準値・同一ペアの指標・時間・モデルidentity・メモリの境界を保存します。正規化CSV、原CSV、出典manifestのSHA-256は別項目です。現在値のメモリ採取とprocess生涯kernel peakを混同しません。

version 2のreportは条件ごとと全体の `scalarFailures` を記録します。数値不一致は試験失敗を維持したまま、残りの原出力とlease検査も記録します。reportが存在するだけでは合格ではなく、試験の正常終了と不一致0件の両方が必要です。入力・identity・shape・ライフサイクル・再実行一致が不成立なら、直ちに試験を停止します。

実モデル、通常Mac/iPhone画面、読み上げ、署名build、本人受入は監督側の確認です。CPU成功や試験入口の追加だけでは合格にしません。

[検証記録](validation/chronos2-bike-20261113/README.md)に、CPU専用の固定5窓を2回照合した結果、独立した指標計算、通常Mac操作、売上/NYC回帰、署名host buildを残しています。実iPhoneの推論と本人受入は未確認です。
