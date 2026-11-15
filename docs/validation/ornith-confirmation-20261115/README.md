# Ornithによる確認文判定の初回試用

2026-11-15、Issue #273。通常Ornith 1.5 35B-A3B MLX BF16をTukuru SFから実行し、`InspectionRules.confirmation`とprivate helperだけを実装した。親は実行前に採点用66ケースと検証器を固定し、結果を保存した後にSwift Testingの回帰と文書を統合した。

## 結果

| 項目 | 結果 |
|---|---|
| 変更前の固定66ケース | 48成功・18失敗（予定した新しい肯定連結の失敗） |
| 初回SF提出の独立検証 | 66成功・0失敗 |
| 親による本体修正・修正依頼 | いずれも0回 |
| worker内のパッチ | 初回実装とコンパイルエラー修正の2イベント |
| workerの所要時間 | 677.699秒（約11分18秒） |
| SF全体の所要時間 | 792.347秒（約13分12秒。workerを含む） |
| 通常設定・元モデル資産 | 前後一致、modelと共有lockを解放 |

workerは`contains`の引数labelが欠けたコンパイルエラーを自己修正した。検証コマンドのパス誤記も再試行している。「初回提出で合格」は、途中の編集や自己検証で失敗しなかったという意味ではない。66ケースはworkerにも見える開発用の固定条件で、未知課題の正答率ではない。

CLI申告はinput 524,952（cached 475,094、差引49,858）、output 28,927（reasoning 28,117）。合計入力にはツール往復による再入力を含む。親・レビュー側のtokenや電力、同じ課題をフロンティアだけで実行した対照は計測していない。今回だけでtoken削減や速度改善を主張しない。

## 検証

固定入力をSwift 6で実際のCoreのsourceへ渡す独立検証と、既存Package全体の`swift test --force-resolved-versions`が成功した。新しいSwift Testingでも同じ66ケースを本番の確認メソッドへ渡し、既定語彙は実際のcatalogから読む。通常suiteの実モデル・実機等のopt-in試験はskipのまま保持する。

AGENTS.mdのprivacy・lint・format・Irodori入力build・Swift試験・Mac Release・pinsを確認した。pinsの最初の失敗は、新規checkoutのXcode build前に実行してDerivedDataが無かったため。版の不一致とは扱わず、build後の成功と元の失敗を両方残す。

SFの初回source・task・oracle・compiler出力・workerログは非公開で保持し、[条件と原記録hash](results.json)に対応付けた。実装後の親レビューと相手レビューはPRの固定base/headで確認する。`verified_pending_review`というSF結果だけをPRの受入に代用しない。

## 残る確認

CONFIRM-05のiPhone上の音声操作は本人確認待ち。今回の自動検証は確認文の純ロジックと既存回帰であり、音声認識や実機操作の受入を代行しない。以後の改善候補は、長いreasoningの制御と、準備・検証を含めた同条件比較。今回の変更へ推論設定の調整は追加していない。
