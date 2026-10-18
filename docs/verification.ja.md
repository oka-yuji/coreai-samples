# 検証と計測

[トップ](../README.ja.md) · [English](verification.md) · [ランタイム](runtime.ja.md)

## 検証

ビルドとモデルを使わない確認の全体を、リポジトリのルートで実行します。Apple シリコンの Mac、macOS 27、`xcode-select` で選んだ Xcode 27 が要ります。

```bash
./check.sh
```

このスクリプトは、`CoreAISamplesKit` をビルドして試験し、`Tools/Verify` をビルドして CPU の試験を Release で実行し、macOS アプリを Release でビルドし、Agent のオフラインの自己テストを実行します。その前に、引き継いだ `COREAI_*`、`GLINER_*`、`DECIDE_*`、`CLEF_*` の設定を消すので、関係のないモデルの試験の設定で推論が始まることはありません。最後の行は、終了コード 0 の `CHECK PASS` か、終了コード 1 の `CHECK FAIL: <stage>` です。

言語モデルの確認を足すには、JSON の結果の検査のために Python 3 を入れ、バンドルのフォルダを `:` で区切って渡します。最初のバンドルは、公式のランタイムとの比較と状態の確認にも使います。Qwen3.5-0.8B のような小さいバンドルなら、この確認は短く済みます。

```bash
COREAI_TEST_BUNDLES=/path/to/bundle-a:/path/to/bundle-b ./check.sh
```

これで、モデルを使う試験、Apple の sequential エンジンと比べる 2 つの固定のプロンプト、自前のランタイムでのチャット 2 ターンとそのあとの処理済みの状態の確認が加わります。2 つのプロンプトは、対応する `prefill-function` の経路で、生成したすべてのトークンと各ステップの logits が一致しなければなりません。2 つの `main-only` の記録は prefill の経路が違うので診断用に残し、その一致は合否に使いません。記録の欠け、報告されたエラー、状態の確認の失敗は、確認のコマンドが正常に終わってもスクリプトを失敗させます。JSON の記録は `.build/check/` に保存されます。

ランタイムとデモアプリの計測は、次の環境で行いました。

| 構成要素 | 計測した構成 |
|---|---|
| macOS | 27.0、ビルド 26A428 |
| Xcode | 27.0、ビルド 27A266a |
| Mac | MacBook Pro(Mac16,6)、Apple M4 Max |
| ユニファイドメモリ | 128 GiB(137,438,953,472 バイト) |

`Tools/Verify` は別の Swift パッケージで、1.0.0 に固定した Apple の `coreai-models` パッケージに依存する唯一の場所です。ライブラリ、CLI、デモアプリは依存しません。その `coreai-verify` は、Apple のランタイム(`--runtime official`、`--engine default` か `sequential`)とこのランタイム(`--runtime own`)を動かし、計測ごとに JSON を 1 行出します。表の「own」はこのリポジトリのランタイムです。

| サブコマンド | すること |
|---|---|
| `chat` | 1 つの LanguageModelSession で複数のターンをストリーミングする。 |
| `tools` | ツールが要るかもしれない質問をし、モデルのステップごとと質問ごとに記録を出す。 |
| `bench` | 読み直しの時間を測り、そのあと固定のプロンプトを、毎回新しいセッションで 3 回ストリーミングする。 |
| `compare` | 自前のエンジンと公式の sequential エンジンに同じトークンを入れ、greedy のトークンと logits を比べる。 |
| `rewind` | 共通の前置きのあとに地点を保存して生成し、復元してもう一度生成する。比較のため、全体の読み直しの時間も測る。 |
| `speed` | 自前のエンジン、公式の sequential エンジン、公式の既定のエンジンを、1 つずつ別のプロセスで交互に動かす。 |

`chat`、`tools`、`bench` は `--runtime` と `--engine` を受け取り、`--runtime own` では `--context-limit` と `--drop-old-tool-outputs` も受け取ります。

```bash
cd Tools/Verify
swift build -c release
.build/release/coreai-verify compare --model <bundle> --max-tokens 64 --prompt "en=List three fruits, one per line."
.build/release/coreai-verify chat --model <bundle> --runtime own --max-tokens 64 --prompt "What is the capital of France? Answer with one word."
```

`CoreAISamplesKit` フォルダで `swift test` を実行すると、使い回しの計画、解析器、サンプラー、文脈の詰め方、RAG の段落と日付のような、モデルの要らない試験が流れます。モデルを読む試験は、`COREAI_TEST_BUNDLES` にバンドルのフォルダを `:` で区切って並べたときだけ流れます。

```bash
cd CoreAISamplesKit
swift test
COREAI_TEST_BUNDLES=/path/to/bundle-a:/path/to/bundle-b swift test
```

モデルを読む試験は 1 つの直列のスイートにまとめてあり、トークナイザだけの確認は独立しています。流す前に、`CLAUDE.md` にある重いジョブのロックを取ります。`seededSamplingRepeats` は、使い回しを止めた状態で、top-k(40)と top-p(0.95)、temperature 0.8、seed 7 の生成したトークン列全体を比べます。完全に再現するには、バンドル、プロンプト、サンプリングの設定、prefill の切り方が同じでなければなりません。`COREAI_SEED_REPETITIONS` は既定の 20 回を変えます。公開の候補の大きいバンドルには、同じ試験を 5 回で使います。

境目の試験は、16,384 トークンの文脈に対して、入力をちょうど 16,382、16,383、16,384、16,385 トークンで描画します。どちらの文脈あふれの方針でも、最初の 2 つは最大 2 トークンと 1 トークンの生成の余地を残し、残りの 2 つは上限と入力の数とともに `contextSizeExceeded` を投げなければなりません。別の履歴で、ツールの結果の切り詰め、そのメタデータ、縮められない文脈あふれを確かめます。状態の確認は、8 目盛りの同点近傍の規則を使って、全体の読み直しと比べます。

`twoSessionsShareOneModel` は 1 つのモデルに 50 組の依頼を送り、executor の呼び出しの重なりを記録し、相手のセッションの印が混ざらないかを確かめ、5 組に 1 組は最初の断片が届いたあとで片方の依頼を取り消します。回復と最後の状態も確かめます。`COREAI_CONCURRENCY_PAIRS=1` で、Thread Sanitizer で流すための 1 組を選びます。これらは明示したときだけ流れる確認で、飛ばしたモデルの試験はバンドルの検証になりません。

```bash
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_SEED_REPETITIONS=5 swift test --filter ModelIntegrationTests.seededSamplingRepeats
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_CONCURRENCY_PAIRS=1 swift test --sanitize=thread --filter ModelIntegrationTests.twoSessionsShareOneModel
```

自前のランタイムの `chat` と `tools` は `--reload-interval` と `--reload-footprint-bytes` を受け取り、0 でそれぞれの条件を止めます。`chat --new-conversation-every 10` は 10 プロンプトごとに新しいセッションを始めます。自前のランタイムのチャットの記録には `generated_tokens` があり、停止のトークンを除いて、サンプリングした出力の ID をすべて含みます(応答の上限に達したときの最後の未処理の ID も含む)。失敗した CLI のターンには、この項目がありません。Verification SPI は、依頼だけの検査を通って実行に入った最後の依頼を返します。デコードの前は空、成功すれば全体、デコードの失敗や取り消しではその時点の途中のサンプルです。依頼だけの検査で断られた依頼は実行のロックの前で止まり、前のスナップショット、処理済みの状態、会話の追跡を変えません。executor が動く前にフレームワークが断った場合も、スナップショットは変わりません。ターンとツールのステップの記録には、読み直しがあれば `reload_seconds` が入り、メモリの記録には graphics footprint のバイト数(graphics footprint の ledger と、その圧縮の ledger の和)が入ります。`chat --check-state` は、読み直した各ターンと最後の状態を確かめます。

```bash
COREAI_TEST_BUNDLES=/path/to/bundle swift test --filter ModelIntegrationTests
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_TEST_RELOAD_INTERVAL=0 COREAI_TEST_RELOAD_FOOTPRINT_BYTES=1 swift test --filter ModelIntegrationTests.twoSessionsShareOneModel
```

閾値を低くした同時の依頼の実行では、少なくとも 1 回の読み直しが観測されなければなりません。CPU の方針の試験と偽のエンジンの試験に加えて、モデルを使う結果が要ります。

メモリの増え方を測るには、30 プロンプトの自前のランタイムの `chat` に `--memory-sample-turn 10 --memory-sample-turn 20 --memory-sample-turn 30` を足します。既定では off です。選んだターンでは、まず元のその場の `turn` の記録を出し、そのあと同じモデルを保ったまま 5 秒間推論せずに待ってから、ターンの番号、`wait_seconds`、メモリを取るまでの実際の `elapsed_seconds`、いつものメモリの項目を持つ別の `memory-sample` の記録を出します。状態の確認を頼んでいれば、そのあとに行い、最後の確認も含みます。重複したターンの番号や存在しないターンの番号は、読み込みの前に断ります。

5 秒の間隔は計測の決まりで、どの端末でもメモリが落ち着いた、回収が終わったことを保証するものではありません。その場の値とピークも、遅らせた標本と並べて残します。footprint の閾値の較正は、引き続きターン直後のその場の footprint の最大を使い、1.5 倍して 10 進の GB に切り上げます。観測の項目と待ちと取得の振る舞いの CPU だけの確認は、`swift test --package-path Tools/Verify` で流れます。

## モデル別の結果

Qwen3.8 27Bの参照との照合結果と未達の基準は[モデル別の検証](qwen3.8-27b.ja.md#検証)、速度とメモリは[主な数値](qwen3.8-27b.ja.md#主な数値)にあります。ほかのモデルは[モデル一覧](../README.ja.md#モデルとロードマップ)から各ガイドを参照してください。
