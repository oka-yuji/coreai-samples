# coreai-samples

## Swift 規約

- 規約は `~/claude-config/conventions/swift/`(`architecture.md`、`testing.md`)。このリポジトリでの上書きは、下の「規約との差分」に書く。
- PR の前に、次をすべて通す:

  ```sh
  swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache
  python3 ~/claude-config/conventions/swift/scripts/check-format.py --base origin/main
  swift test --package-path CoreAISamplesKit
  xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release -destination 'platform=macOS' build
  ```

- 大きさの合図(合否には使わない): `swiftlint lint --config .swiftlint-size.yml --quiet`

### 規約との差分

- T7(`@Test` の表示名は日本語の 1 文): 英語の 1 文で書く。理由: 下の「公開物の規約」で、ソースに日本語を入れないため。代わりの判定: 表示名が Issue の `[test]` の文を英語にしたものか、レビューで見る。
- T3 と `ADOPTION.md` §4(汎用 Store): `Store.swift` はアプリの target ではなく、`Feature.swift` と同じ `CoreAISamplesKit/Sources/FeatureStore`(既定分離 nonisolated。`Store` は `@MainActor` を明記している)に置く。テンプレートのコメントとテストの表示名は英語に訳し、コードはテンプレートと同じにしてある。テンプレートが変わったら、コードの差分だけを取り込む。lint の純ロジックの規則は `FeatureStore` にも掛け、import と並行処理の規則からだけ `Store.swift` を外す。代わりの判定: `swift test --package-path CoreAISamplesKit --filter StoreTests`。

## 公開物の規約

- コード内のコメントは英語で書く。
- ソースに日本語の文字列を入れない。画面の表示は英語にし、例の資料は JSON の資源に置く。
- コミットと PR に `Co-Authored-By` を付けない。

## 重いジョブ

- モデルを読むジョブ(変換、計測、モデルを読むテスト、デモアプリでモデルを動かすこと)は、`mkdir ~/Desktop/SelfApps/CoreLLM/.locks/heavy-job` でロックを取ってから始める。中に `owner.txt`(セッション名、ジョブ名、開始時刻、見込み時間)を置く。取る前に、モデルを動かすプロセス(`coreai-chat`、`coreai-verify`、デモアプリなど)が残っていないことを `pgrep` で確かめる。`mkdir` が失敗したら、持ち主のジョブが終わるまで待つ。終わったら外す(名前を `heavy-job-done-<ジョブ名>-<日時>` に変える)。
- GPU を使う統合テストは `.enabled(if:)` で既定の実行から外す。`swift test` だけでは、モデルを読むテストは動かない。動かすのは、バンドルのフォルダを `:` 区切りで渡したときだけ(このときはロックを取る): `COREAI_TEST_BUNDLES=/path/to/bundle-a:/path/to/bundle-b swift test --package-path CoreAISamplesKit`

- Decide の統合テストは言語モデル用と混ぜず、GPU ロックの下で専用の変数と filter を使う:
  `COREAI_DECISION_BUNDLE=/path/to/decision-bundle swift test --package-path CoreAISamplesKit -c release --filter DecisionInferenceTests`
  20 文書のメモリ試験は `DECIDE_MEMORY_TEST=1` を追加する。測定の JSON は `DECIDE_METRICS_DIR=/path/to/output` で保存できる。
  `COREAI_TEST_BUNDLES` は言語モデル用だけに使い、両変数を同時に設定しない。
- 同じ文書で質問数を変えるメモリ検査は `DECIDE_SHAPE_MEMORY=1` と `--filter DecisionInferenceTests.questionSizeMemory`。headless の文書再利用経路は `--decide-json input.json output.json --reuse-document`。
