# #92 UI 独立レビュー

対象: base `45936ba8f50e9d59182dae5d5948c232ba218fee` に対する未コミット・未追跡差分。担当はSpeechFeature、UI、保存/選択/Assembly、ナビゲーションとビルド設定。

## 結果

確定欠陥・plausible残件なし。候補の個別反証と確認範囲は `review.json`。

- session/phaseとStore tokenの二段で古い生成・再生・読込通知を拒否。
- stop/endedはstoppingを維持し、資源解放後の対応するstoppedでのみidleへ戻る。
- 保存はticket IDとsessionを照合し、パネルに渡すDataとticketは値として固定。
- 起動時にモデルを読まず、保存パスはOpen last modelボタンで明示的に開く。
- 生成pbxprojをbaseと構造比較。MacへSwift4ファイル・JSON1・Speech製品3を追加。既存製品の削除なし。iOSのsources/resources/productsは不変。
- 手書きの変更Swiftに禁止されたコメント・日本語文字列なし。例文はJSON資源。

## 検証

Core AIを含まない `/tmp/irodori-speech92-review/PureReview` で原文のSpeechFeatureTests 7件を実行し通過。ログは `pure-tests.log`。

親の最終保存ログで、Speech CPU27件、Audio/Service TSan9件、XcodeGen版Releaseアプリbuild、lint0/formatの通過を確認した。これらは本担当が再実行したものではない。後から通知されたSpeechRunner.stopの取消判定と追加cancelledStopテストも読解確認済み。新規サービス試験は独立packageの7件には含まれない。

## 変更監査

開始後に担当Feature/View/設定の変更なし。通知されたSpeechRunnerとSpeechServiceTestsの差分のみ追加読解。docs追加は親の継続作業として分離した。`source-scope.json` は当初の18 source/config/resource/testファイルを対象とする最終監査。

## 未確認

U6: 実モデル/GPU、音声、実保存・モデル選択パネル、キーボード/フォーカス、視覚レイアウト。ユーザーが手動確認する。アプリ起動・GUI操作・モデル推論・リポジトリ編集は実施していない。レビュー終了をU6や開発サイクル完了と扱わない。

## ガイドへの追記候補

追記候補なし。
