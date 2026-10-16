# #94 iOS runtime platform/build independent review

Base: `043cf1f2439bcf2912673530d5a9dfd673c3b3e5`。未コミット・未追跡差分の担当範囲を確認。確定欠陥・plausible残件なし。個別の候補・反証・範囲はreview.json。

## 確認結果

- macOS/iOSでRust archive検索先が分離され、実iPhone probeのリンクコマンドにもiOS側パスが使われた。
- archive構成要素のplatformはMacとiOSで混在しない。生成frameworkはIOS/minos27.0で7個のC ABIシンボルを含む。
- iPhoneOS27 SDKでCore AIの特殊化・EFR・実行APIの存在を確認。
- MacはGPU優先/EFR trueを維持。iOSはmanifest順のtext/durationがfalse、DiT/codecがtrue。数式・ノイズ・固定入力レイアウトはbaseから変化していない。
- build scriptを/tmpへ複製しCargoだけstub化。既定Mac、iPhone環境、引数優先、Simulator拒否を確認。元scriptや既存worktreeを変更しない。
- 親のCPU31件・iPhone framework最終build成功ログを読解確認。これは本担当が再実行したテストではない。

## 制限と未確認

対象は物理iPhone。Simulatorは現在のbuild scriptで明示拒否され、device archiveをSimulator用とは扱わない。Catalystも未検証。

実機OSは本担当で確認していない。Core AIはiOS27必須。iPhone上の特殊化・動的形状・最大メモリ・CPUノイズ/音声一致は未実行で、後続#95へ残る。Macでの合格をiPhoneでの数値一致へ外挿しない。

PlaybackSessionの通知処理・所有権詳細は他担当の範囲。初回probeのiOS27非推奨通知warningは親の変更後の最終probeで解消した。

## ガイドへの追記候補

- 2026-10-13 Rust tokenizers0.22.2/unicode-normalization0.1.22のaarch64-apple-ios static libraryをiOS27向けSwift SpeechRuntime frameworkへリンクでき、生成frameworkのLC_BUILD_VERSIONはIOS/minos27.0、7個のC ABIシンボルも含まれた。モデル実行やiPhone数値一致を確認した結果ではない。[実測] coreai-samples #94、/tmp/irodori-ios94-iphone-link-final.log、/tmp/irodori-runtime94-review/iphone-framework.json
