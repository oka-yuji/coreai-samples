# #95 iPhone sampler independent review

Base: `0894efabd9b793efc9db0a57b581db93966a6511`。未コミット・未追跡を含むUI/library/feature/target設定の独立レビュー。確定欠陥1件を修正後に再確認し、残るconfirmed/plausibleはない。

## 修正済み P95-01（P2）

初回の署名Release生成物は`LSSupportsOpeningDocumentsInPlace=true`だが`UIFileSharingEnabled`が欠落。project.ymlのビルド設定だけでは反映されず、通常ユーザーのFiles→Documentsモデル配置に必要な宣言が揃わなかった。親も実物で再現。

修正後は`SpeechPhone/Info.plist`に両キーをtrueで明示し、project.ymlと生成targetが同ファイルを参照。ResourcesへInfo.plistを重複コピーせず、`irodori-ios95-signed-final.log`はBUILD SUCCEEDED。生成物の両キーtrue、bundle IDとminOS27.0を再確認した。

[Apple公式のFiles公開条件](https://developer.apple.com/library/archive/documentation/General/Reference/InfoPlistKeyReference/Articles/LaunchServicesKeys.html)。修正前後のplist観測はsharing-initial-observation.json／sharing-fixed-observation.json。

## その他の候補と反証

- 通常起動はlibrary一覧だけ。推論は生成操作、診断は`--speech-check-run`指定時だけ。schemeに自動診断引数なし。
- library取得結果はsessionとpresented、check結果はsessionとphaseで古い結果を拒否。background/disappearで終了し、資源停止待ちの状態を保持する。
- listingは隠し項目・symlinkを除外。壊れたsymlinkと有効modelの共存も独立CPU試験で通過。
- Mac/Extractのtarget構造をbase/currentで比較し、sources/resources/products/settingsに変更なし。SpeechPhoneのID・署名設定は独立。
- Swift6/MainActor/upcoming2を実コンパイラ引数でも確認。
- increased-memory-limit=trueはsource、xcent、Mach-O署名XML、profile内XMLで一致。コード設定の欠落候補は反証。
- 追加されたSpeechChecksとモデル名の衝突防止を読み、対応テストも再実行した。

個別台帳と完全な確認範囲はreview.json。

## CPU検証

CoreAIを含まない一時packageで8件通過（Feature3、Library/Files4、追加dangling symlink1）。ログはcpu-tests-final.log。親のCPU45件と署名buildの保存ログも確認したが、それらは本担当が再実行したものではない。

## 未確認・処理制約

端末OS、install受理、Files実画面、ネイティブ保存/再生、GPU特殊化・音声・メモリは未実施。sandboxのcodesign信頼検査はCSSMERR_TP_NOT_TRUSTED、security CMSデコードも失敗した。署名/profile内容の構造確認は信頼チェーンやOS受理の保証ではない。アプリ起動・GUI・GPU・実機操作・リポジトリ編集は行っていない。

## ガイドへの追記候補

- 2026-10-13 Xcode27のSpeechPhoneではINFOPLIST_KEY_UIFileSharingEnabled=YESでも生成Info.plistにキーが出ず、Files公開に必要な両キーが揃わなかった。専用Info.plistにUIFileSharingEnabledとLSSupportsOpeningDocumentsInPlaceをtrueで書き、INFOPLIST_FILEで指定すると署名Release生成物にも両方trueで入った。[実測][公式] coreai-samples #95、sharing-{initial,fixed}-observation.json、Apple InfoPlistKeyReference LaunchServicesKeys
