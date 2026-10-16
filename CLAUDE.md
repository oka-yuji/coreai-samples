# coreai-samples

## Swift 規約

- 規約は `~/claude-config/conventions/swift/`(`architecture.md`、`testing.md`)。このリポジトリでの上書きは、下の「規約との差分」に書く。
- PR の前に、次をすべて通す:

  ```sh
  swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache
  python3 Tools/Format/check_format.py --base origin/main
  ./Tools/Irodori/build-inputs.sh
  swift test --package-path CoreAISamplesKit
  xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release -destination 'platform=macOS' build
  ```

- 大きさの合図(合否には使わない): `swiftlint lint --config .swiftlint-size.yml --quiet`

### 規約との差分

- 書式の命名規則: `WebSearchArguments.max_results` だけを固定された外部スキーマ名の例外とする。名前を変えると `web_search` の引数と凍結した比較条件が変わるため。`Tools/Format/check_format.py` は共通の厳格な検査をそのまま実行し、この型・宣言・診断の1件だけなら例外として扱う。ほかの診断、同名の追加宣言、型の変更、検査自体の失敗は通さない。`--agent-selftest --offline` の `tools/definitions` で外部名と順序を照合する。

- T7(`@Test` の表示名は日本語の 1 文): 英語の 1 文で書く。理由: 下の「公開物の規約」で、ソースに日本語を入れないため。代わりの判定: 表示名が Issue の `[test]` の文を英語にしたものか、レビューで見る。
- T3 と `ADOPTION.md` §4(汎用 Store): `Store.swift` はアプリの target ではなく、`Feature.swift` と同じ `CoreAISamplesKit/Sources/FeatureStore`(既定分離 nonisolated。`Store` は `@MainActor` を明記している)に置く。テンプレートのコメントは除き、テストの表示名は英語に訳す。コメント以外のコードはテンプレートと同じに保ち、更新時も実装の差分を取り込む。lint の純ロジックの規則は `FeatureStore` にも掛け、import と並行処理の規則からだけ `Store.swift` を外す。代わりの判定: `swift test --package-path CoreAISamplesKit --filter StoreTests`。


- A12(安全条件を隣のコメントに書く): 公開ソースにコメントを置かないため、条件と検査を下の表に記録する。`concurrency_escape` の検出は維持し、新しい宣言を自動的に許可しない。既存の宣言の基準値は `.swiftlint-baseline.json` で追跡する。変更時は新しいアクセス経路も条件を満たすか確認する。
- lint の抑制をソースのコメントに書かない。`no_inline_swiftlint_commands` と公開ソースの監査で確認する。基準値の更新は、ruleIdentifier・ファイル・該当コード・件数を変更前と比較し、許可の追加を説明する。既存ファイル全体を新しく除外しない。
- T1(新しい試験は Swift Testing): Agent の既定、カセット、取得先の制限、Hub の照合のアプリ統合確認は既存の `--agent-selftest` に追加できる。アプリに Swift Testing の target がないため。新しい独立した純ロジックの単体試験は Swift Testing に置く。アプリの試験 target を導入したら統合確認の移設を検討する。

### 既存の並行処理の安全条件

| 宣言 | 所有者と守る条件 | 確認方法 |
|---|---|---|
| `EngineBox` (`CoreAIBackend/LocalLanguageModel.swift`) | `Runtime` だけが箱を所有する。Engine と checkpoint の可変状態へのアクセス、状態の照合、非同期の推論は同じ Runtime の lock の内側に置き、await 中も他の依頼を入れない。初期化済みの不変な設定だけを lock の外で読む。 | `sessionReusesStopsSamplesAndCancels`、`sessionCallsToolsReusesAndRecovers`、`StateCheck` の既存の検証。GPU を読む試験は下記のロックと環境変数で実行する。 |
| `HubDownloader` (`Demos/ModelManager/HubDownloader.swift`) | ダウンロードごとに新しいインスタンスを使う。`fileList` が session を初期化してからファイルの取得を並行化する。同じインスタンスの download / fileList を別々に同時実行しない。entries・liveTasks・didCancel と continuation の完了権の読み書きは NSLock で保護し、再開と利用者の callback は lock の外で行う。 | 呼び出し元と全 delegate / cancel 経路のレビュー。Hub の照合を変更する Issue では取消・完了・再開の自己テストも確認する。 |
| `ProgressTracker` (`Demos/ModelManager/HubDownloader.swift`) | totalBytes と startTime は不変。completedBytes と lastReportedAt の読み書きは NSLock の内側で行い、lock の外では取り出した値だけを使う。 | add(bytes:) の全アクセスのレビューと、ダウンロードの進捗・再開の確認。 |
| `ExtractEval.run()` の既存の Task (`Demos/Extract/ExtractEval.swift`) | ヘッドレス入口が一度だけ MainActor 上で execute を開始し、完了時に終了コードを返してプロセスを終える。UI の非同期状態管理の例外には広げない。 | `--extract-eval` の dev50 の出力を既存の基準と照合する。既存の行の抑制を baseline へ移したもので、新しい Task の許可は追加しない。 |
| `ObjectFinderPreview` (`ObjectFinderRuntime/ObjectFinderPreview.swift`) | Session と preview layer の結び付けは MainActor の初期化で一度だけ行う。Session の configure/start/stop は一つの ObjectFinderCapture serial actor に限定する。UI は session を操作せず、layer の配置・描画だけを MainActor で行う。preview connection の向きは UI の最新姿勢から MainActor で設定し、capture 再接続後にも現在値を適用する（Apple 公式の MainActor preview / capture actor 分離に準拠）。callback は世代付きの不変な矩形だけを MainActor へ渡し、古い世代を拒否する。 | 30 fps 設定・capture の全アクセス経路・layer の全呼出元のレビュー、署名 iOS build、ObjectFinderTests。Apple AVCam の preview/session 接続と SDK27 宣言を照合。 |


## 公開物の規約

- 手書きのソース・テスト・設定には、doc コメントを含めてコメントを書かない。SwiftPM の `swift-tools-version` と shell の先頭 shebang は、ツールが解釈する宣言として保持する。
- ソースに日本語の文字列を入れない。画面の日英翻訳はString Catalog、例の資料は JSON の資源に置く。`RAGContext.swift` の `dateLabel`、`instruction(today:)`、`block(_:number:)`、`message(question:sources:)` の既存のプロンプト定数だけを例外とする。`noSourcesReply` は例外に含めない。表示言語の切替で入力やモデルのプロンプトを変更しない(2026-10-09のユーザー指定)。
- コミットと PR に `Co-Authored-By` を付けない。

## 重いジョブ

- モデルを読むジョブ(変換、計測、モデルを読むテスト、デモアプリでモデルを動かすこと)は、`mkdir -p ~/Library/Caches/coreai-samples && mkdir ~/Library/Caches/coreai-samples/heavy-job` でロックを取ってから始める。中に `owner.txt`(セッション名、ジョブ名、開始時刻、見込み時間)を置く。取る前に、モデルを動かすプロセス(`coreai-chat`、`coreai-verify`、デモアプリなど)が残っていないことを `pgrep` で確かめる。`mkdir` が失敗したら、持ち主のジョブが終わるまで待つ。終わったら外す(名前を `heavy-job-done-<ジョブ名>-<日時>` に変える)。
- GPU を使う統合テストは `.enabled(if:)` で既定の実行から外す。`swift test` だけでは、モデルを読むテストは動かない。動かすのは、バンドルのフォルダを `:` 区切りで渡したときだけ(このときはロックを取る): `COREAI_TEST_BUNDLES=/path/to/bundle-a:/path/to/bundle-b swift test --package-path CoreAISamplesKit`

- Decide の統合テストは言語モデル用と混ぜず、GPU ロックの下で専用の変数と filter を使う:
  `COREAI_DECISION_BUNDLE=/path/to/decision-bundle swift test --package-path CoreAISamplesKit -c release --filter DecisionInferenceTests`
  20 文書のメモリ試験は `DECIDE_MEMORY_TEST=1` を追加する。測定の JSON は `DECIDE_METRICS_DIR=/path/to/output` で保存できる。
  `COREAI_TEST_BUNDLES` は言語モデル用だけに使い、両変数を同時に設定しない。
- 同じ文書で質問数を変えるメモリ検査は `DECIDE_SHAPE_MEMORY=1` と `--filter DecisionInferenceTests.questionSizeMemory`。headless の文書再利用経路は `--decide-json input.json output.json --reuse-document`。

- EmbeddingGemma 2のCPU検査は既定の`swift test`に含む。`EG2_TOKENIZER=/path/to/bundle`を渡すと参照12入力のtoken列も照合する（モデルは読まない）。実モデルを読む`embedding-verify`と画像検索デモは共有ロック下で動かす。入力契約とコマンドは`docs/embeddinggemma2.md`。

## 変換の保守（2026-10-15）

- Core AIの今後の開発・変換ツールの正本はこのリポジトリ。CoreLLMは過去の記録として保管し、新しいコード・計測を追加しない。
- 変換の保守用入口は `python3 -m Tools.Conversion.local setup`、`plan`、`run`。手順は `Tools/Conversion/README.md`。一般向けの安定APIや全モデル対応を約束する製品ではない。
- 個人用の設定・依存ツール・モデル・生ログ・実行キャッシュは `.local/conversion/` に置き、Gitへ追加しない。共通処理とテストは履歴管理する。モデルの重みは従来どおり別配布する。
- 変換と参照計算は上記と同じ機械内のheavy-jobロックを使う。`COREAI_HEAVY_JOB_LOCK`で変更する場合は、このMacで動く全モデルジョブの取得先をそろえる。移行前から動いているジョブの終了を確認してから新しい場所を使い、既存ロックを削除・奪取しない。
- GLiNERの任意fixture試験は `GLINER_FIXTURES_DIR` で参照データの場所を指定できる。既定は `~/Library/Caches/coreai-samples/reference/models/coreai/gliner`。以前のラボを自動検索しない。fixture内の相対パスで参照する `reference/docs/results/` の構造も保つ。
- 公開用の数値記録は `python3 -m Tools.Conversion.summary <report.json>` で抽出し、公開前に内容を確認する。生のreport・manifest・process.logには環境変数・コマンド・入力本文が入り得るため、そのままコミットしない。
- 変換コードを変更した場合は専用環境で `python -m unittest discover -s Tools/Conversion/tests -v` も実行する。NumPy未導入でスキップされた数値試験を合格として扱わない。

## Irodori runtime

- The Mac and iOS 27 speech runtime uses the pinned Rust input adapter in `Vendor/IrodoriInputKit`. Build it with `Tools/Irodori/build-inputs.sh` before package tests. These operations do not execute Core AI models.
- `SpeechCore` and `SpeechFeature` are pure-logic targets. `SpeechRuntime` owns a single in-flight operation and holds its busy gate across suspension. A release waits for that operation to finish; it does not clear a newer pipeline after resuming.
- Test display names and source follow the existing English-only/no-comments rules. Model generation and listening in the app are user acceptance checks for this change.
- `SpeechFeature` owns session and save-panel tickets. `Store` owns task cancellation; `SpeechRunner`/`LiveSpeechService` join runtime release before acknowledging stop. `AudioPlayback` resolves its continuation after synchronous stop instead of waiting for a delegate completion callback. CPU tests use fake devices.

- For a physical iPhone, install `aarch64-apple-ios` with rustup and run `Tools/Irodori/build-inputs.sh iphoneos`. Mac and iOS archives stay separate. Simulator support is not provided. iOS audio activation/deactivation uses the asynchronous SDK27 API; stop acknowledgement waits for deactivation. External audio interruption is distinct from Task cancellation.

- `SpeechPhone` is a separate iOS 27 target/bundle ID. `SpeechLibraryFeature` and `SpeechCheckFeature` also use the shared Store; neither starts model inference on ordinary launch. `--speech-check-run` explicitly selects diagnostics. Device checks must use the requested iPhone, protect existing model/output directories, and record OS/connectivity separately from host build success.
- For Phone changes, also build the signed `SpeechPhone` product and inspect the actual Info.plist for both file-sharing keys, plus the memory entitlement. The source settings alone are not evidence that the generated app contains those keys.
