# Chronos-2による商品売上予測

[English](chronos2-sales.md)

MacのサイドバーとiOSのModelsで、Chronos-2の「商品売上予測」を開きます。既存のFP32資産、取得・準備画面、所有者の許可、leaseを共用します。「公開売上データを読み込む」で取得し、商品・Europe/Romeの予測開始日・7日または14日を選んで「この端末で予測」を押します。表示・取込・モデル準備・選択変更だけでは、取得も予測も始まりません。モデルなしでもデータを閲覧できますが、予測結果を作ったり、推論失敗を別の結果へ置き換えたりしません。

## 出典と条件

作成者はPaolo Mancuso、Veronica Piccialli、Antonio M. Sudoso（2021年）です。

原典は[Hierarchical Sales Data、Mendeley Data v1](https://doi.org/10.17632/njdkntcpc9.1)です。[UCI](https://archive.ics.uci.edu/dataset/611/hierarchical+sales+data)の表示はCC BY 4.0ですが、Mendeleyの原配布は**CC BY-NC 3.0**です。原典の非商用利用条件を適用し、取得ボタンの隣にも明示します。実売上・元CSV・派生fixtureはアプリやリポジトリへ同梱しません。

取得先は[原典ファイル](https://data.mendeley.com/public-files/datasets/njdkntcpc9/files/08bb4f43-6dfa-4995-b268-42fa0690ba6b/file_downloaded)だけです。応答上限2 MiB、888,985 bytes、SHA-256 `0dfd4a5bf801bf79ddb5fb1f6bc8d023487c36dd6a2a70b312cdfa9fa6568d83`を検査してから解析・保存します。標準の配布元リダイレクトは許しますが、内容の一致は必須です。actorがアプリの専用cacheだけを管理し、読込ごとに再検査します。有効なcacheはオフラインでも使えます。破損は画面に表示し、再度の明示読込で再取得します。[再取得手順](../Tools/Forecasting/README.md)。

## 入力と比較

引用符付きCSVを解析し、DATE・118個のQTY・対応するPROMOの全237列を検査します。不正な引用符・列数・ヘッダー・重複/逆順/不正日付・非有限/範囲外/負/小数の数量・0/1以外の販促を拒否します。商品は予測成績に関係なくB1_1、B2_1、B3_1、B4_1へ固定します。販促は検査のみで入力へ使いません。

2014-01-01〜2018-12-31のEurope/Romeの1,826暦日に、原典1,798行を配置します。先頭を含む28欠損日をnilで残し、ゼロや休業と仮定しません。DSTをまたぐ日付更新に86,400秒の加算を使いません。軸・日付ごとの読み上げも端末のtimezoneによらずEurope/Romeです。

入力は開始日より厳密に前の365暦日です。選択できる開始日は、全予測期間が原典内に収まる範囲です。季節ナイーブは開始前の最後の7暦日だけを使用し、14日では同じ週を2回繰り返します。2週目の基準値に将来実績を使いません。実績と基準値の欠損はnil、モデル入力の欠損はNaNです。ChronosContextが365点の左側に3点のNaNを追加します。

モデルは既存の単系列FP32・21分位×16点、公式revision `29ec3766d36d6f73f0696f85560a422f50e8498c`です。全336原値、実資産identity、入力、実行時間とその計測境界を保持します。表示はP10/index 2、中央値/index 10、P90/index 18の先頭7/14点です。負値は表示・指標のみゼロへ丸めます。履歴は点、実績は四角で欠損をつながず、基準線も欠損で分けます。

MAE/WAPEは実績・中央値・基準値の揃う同じ日だけを使い、比較件数を表示します。WAPEの分母はその日の実績合計で、ゼロなら未定義です。有効日がなければMAEも未定義です。実売上は潜在需要ではなく、P10–P90の的中率や商用品質を保証しません。28日・共変量・再帰延長は未対応です。

## 停止と検証

ForecastFeatureが純粋なデータ・入力・指標・ticket状態、StoreがTask、Runnerが副作用と終了待ち、MainActorのServiceが境界、データactorがI/Oを所有します。Serviceは要求が読込済みデータの許可された切り出しと一致すること、結果identityがModels資源と一致することを検査します。sessionごとにService/ChronosRuntimeを作ります。停止は許可失効→取消→処理join→runtime解放→Modelsのlease返却です。取消を無視する処理や古い完了・closeが新しいsessionへ作用しないことをCPUで検査します。

既定試験は合成データによる解析・暦日・欠損・未来漏洩・指標・状態・停止/再試行・cache・所有権を扱い、通信やモデル実行を行いません。実モデル検査は[英語版のコマンドと参照形式](chronos2-sales.md#ownership-and-verification)に従い、共有heavy-jobロックと4つの環境変数を明示して親担当者が実行します。独立した公式FP32参照を複数商品・日付で用意し、本番Models/Store/Service/Sessionsで全入力・全21×16出力をatol/rtol各1e-4、identity、close後lease 0、再表示後の再実行まで照合します。既存reportへの上書きは拒否します。

reportには実データ由来の入力・全出力も含むため非公開で保存します。source/reference hash、同一ペアの指標、各要求の時間、OS/機種/precision、現在のRSS/footprintとprocess生涯のkernel peakを区別します。SDKの返却件数で項目の有効性を確認し、未対応のメモリ値はnullです。runtime時間は必要な検証・コピー・モデル読込を含み、データ取得・Models取込・描画・指標計算を除きます。生涯peakを要求単位のpeakやGPU総量へ読み替えません。

[検証記録](validation/chronos2-sales-20261113/README.md)には、固定16条件を2回実行した全分位照合、独立した指標計算、通常Macの操作、署名済みのhost buildを残しています。実iPhoneの推論と本人受入は未確認です。固定バックテストと機能確認中の時間を、一般的な予測品質や制御した性能benchmarkへ読み替えません。
