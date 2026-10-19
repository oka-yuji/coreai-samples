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

## 計測項目と境界

依頼ごとの計測は、明示したときだけ行います。`chat --runtime own --measure summary` は各 `turn` の記録に `measurement` を足し、`--measure trace` はさらにサンプルごとの行を足します。既定の `--measure off` では、この項目は無く、生成、停止、エラー、メタデータ、使用量は、この指定が無いときと同じです。`tools` は `--measure` を断り、`bench` にはこの指定がありません。値は `CoreAIBackend` が採り、`coreai-verify` が出力します。Verification SPI の `lastRequestMeasurement()` は、`lastGeneratedTokens()` と同じく実行に入った最後の依頼の計測を返し、その依頼が生成の前に終わったとき(文脈あふれや読み込みの失敗など)は何も返しません。`turn` の記録は、そのターンの依頼に計測が無ければ `measurement` を持ちません。実行の前に断られた依頼では SPI に前の依頼の計測が残るので、CLI は `request_id` を前のターンのものと比べます。時間は 1 つの単調な時計で測った秒、回数は整数です。

ランタイムの生成は、依頼の検査、テンプレートの描画、文脈の詰め方、読み直しの後に始まります。読み直しの時間は別に `reload_seconds` に入ります。読み込みの後の段階は切れ目なく続くので、完了した依頼では、丸めの誤差を除いて `read_seconds` と 5 つの段階の合計の和が `total_seconds` に一致します。エラーで終わった依頼では、その時に進んでいた段階の時間は `total_seconds` だけに入ります。

| 項目 | 開始 → 終了 | 数える対象 | 単位 |
|---|---|---|---|
| `Engine.Stats.decodeSeconds` | 各 `step`: forward の開始(入力の準備を含む)→ `function.run` の後の forward の終了 | 完了した step(`decodeTokens`) | s |
| `Engine.Stats.prepareSeconds` | prefill の各チャンクと各 step の forward: 開始 → `function.run` の直前 | forward | s |
| メタデータの `prefillSeconds` = `read_seconds` | 生成の開始 → 読み込み(prefill または復元)の終了。読み込みが失敗したときは失敗まで | 読んだトークンの `prefilled_tokens`(メタデータの `prefillTokens`) | s |
| メタデータの `decodeSeconds` | 読み込みの終了 → 最後の送信(drain)の終了(サンプリング、文字列化、送信、step をまとめた区間) | — | s |
| `first_token_seconds`、`last_token_seconds` | 生成の開始 → 最初と最後の `sampler.next` が返った時点(停止のトークンのサンプルを含む) | — | s |
| `first_send_seconds` | 生成の開始 → 1 件以上のイベントを送った最初の `output.send` が返った時点 | — | s |
| `sample_seconds` | 前の段階の終わり(最初のサンプルでは読み込みの終わり)→ `sampler.next` が返るまで(取り消しの確認を含む) | 停止のトークンを含むサンプル | s |
| `text_seconds` | サンプルの終わり → `tokenizer.decode` と `parser.feed` の終わり | 停止のトークン以外のサンプル | s |
| `send_seconds` | 文字列化の終わり → `output.send` が返るまで(送るものが無い呼び出しも含む) | 送信の呼び出し | s |
| `step_seconds` | 送信の終わり → 共有の計算で行う `engine.step` が返るまで(その入力の準備と forward を含む) | 完了した step の `steps` | s |
| `prepare_seconds` | デコードのループの前後での `Engine.Stats.prepareSeconds` の差。`step_seconds` の一部 | step の forward | s |
| `drain_seconds` | 最後の段階の終わり → 残りの文字列化と `output.send(last: true)` が返るまで。エラーで終わった依頼では 0 | — | s |
| `total_seconds` | 生成の開始 → drain の終わり(メタデータを送る前)。エラーのときはエラーまで | — | s |
| speed の `load_s` | 開始 → `Engine.init`(own)または `EngineFactory.createEngine`(official。`warmup(queryLength: 1)` はもう含まない)の終わり | — | s |
| speed の `warmup_s` | ロードの終わり → 計測する実行の開始。own は 16 トークンの前置きのあとの 8 トークンの生成と `reset`。official は `warmup(queryLength: 1)`、8 トークンの drain、`reset(to: 0)` | — | s |
| speed の `specialize_s`、`cached` | own の `Engine.LoadStats` の特殊化の時間。キャッシュ済みのモデルでは 0 | — | s |
| speed の `max_tokens`、`generated_tokens` | 要求した上限(以前の `decode_tokens`)と、`onToken` または公式の出力列で実際に受け取ったトークンの数 | トークン | トークン |
| speed の `prefill_s`、`decode_s` | 実行の開始 → 最初のトークン、最初のトークン → 最後のトークン | — | s |
| speed の `decode_tok_s` | `(generated_tokens − 1) / decode_s`。2 トークン未満では無い | トークン | トークン/s |

回数は、`prompt_tokens`、`prefilled_tokens`(この依頼でエンジンが読んだトークン。プロンプトをすべて使い回したときや、読む前に失敗したときは 0)、`sampled_tokens`(`generated_tokens` と同じ並び。停止のトークンを含まず、上限に達したときの最後の未処理のトークンを含む)、`steps`、`sends`(1 件以上のイベントを送った送信。drain を含む)です。`outcome` は `stopToken`、`maxTokens`、`cancelled`、`failed` のどれかで、後の 2 つには `error` が付きます。speed の記録の `stop_reason` は、`generated_tokens` が `max_tokens` に達したとき `maxTokens`、それ以外は `stopToken` です。ここでは自前のエンジンは停止のトークンを無視するので、上限より前に終わり得るのは公式のエンジンだけです。`trace` の行は `sampler.next` ごとに 1 行あるので、`stopToken` の依頼では `sampled_tokens + 1` 行になります。停止のトークンの行には `sample_seconds` だけがあり、後に step が無かった行には `step_seconds` がありません。

[共通の記録の規約](benchmarking.ja.md)の項目には、次のように対応します。

| 共通の項目 | この記録での出どころ |
|---|---|
| `ttft_s` | `measurement.first_send_seconds`。Foundation Models へ最初の出力を渡した時点です。`first_token_seconds` との差は、文字列化と送信の時間です。起点は生成の開始で、`turn` の記録の `ttft_s` はセッションを呼んだ時点からの全体の時間です。 |
| `decode_tps` | `(サンプル数 − 1) / (last_token_seconds − first_token_seconds)`。サンプル数は `stopToken` では `sampled_tokens + 1`、それ以外では `sampled_tokens`(trace の行数)です。speed の実行では `decode_tok_s`。 |
| `latency_s` | `measurement.total_seconds` |
| `first_load_s`、`warm_load_s` | `cached` が false または true の speed の `load_s`(またはチャットの `load` の記録の `init_s`)。自前のランタイムだけです。 |
| `peak_rss_bytes` | speed の `peak_rss_bytes`。子プロセスの生涯のピークを `getrusage` で取ったものです。 |
| `resident_bytes` | `memory.rss_bytes`。`captured_at` の時点の標本です。 |

次の項目は、これらのコマンドでは観測できないか、測っていません。

| 項目 | 状態 |
|---|---|
| GPU の実行時間 | 採っていません。Instruments で観測したときだけ共通の記録に加えます。wall time から推定しません。 |
| 実際の演算の配置 | 返す公開 API がありません。`observed_compute` は常に `unknown` です。own のロードと speed の記録の `requested_compute` と `expect_frequent_reshapes` は、特殊化で要求した設定にすぎません。 |
| 公式の特殊化の時間とキャッシュの命中 | `EngineFactory` からは観測できません。official の speed の記録には `specialize_s`、`cached`、`requested_compute` がありません。キャッシュの状態は手順で固定します。 |
| `peak_footprint_bytes` | 測っていません。`footprint_bytes` は標本で、ピークではありません。 |
| `os_proc_available_memory` | macOS の SDK では使えないため、記録しません。 |

どの `memory` にも、`status`(`ok` か `failed`)、`captured_at`(ISO 8601 の wall clock)、`unit`(`B`)があります。`error`(例: `task_info returned 5`)は失敗したときだけ、`rss_bytes`、`footprint_bytes`、`graphics_footprint_bytes` は成功したときだけ出ます。speed の記録は、`getrusage` が失敗すると `peak_rss_bytes` を出しません。まとめの記録は、成功した取得の最大とその件数 `peak_rss_runs`、速度のある実行の中央値とその件数 `decode_runs` を出します。欠けた値を 0 として数えません。

`summary` は合計と回数を採ります。計測はサンプルごとに最大 4 回時計を読み、`trace` はさらに行を保存します。結果を比べる前に、同じコマンドを `--measure off` と `--measure trace` で交互に流す A/A の確認をし、`turn` の記録の `total_s` と `decode_tok_s` を比べます。

```bash
.build/release/coreai-verify chat --model <bundle> --runtime own --reasoning off --measure trace --max-tokens 32 --prompt "List three fruits, one per line."
```

## モデル別の結果

Qwen3.8 27Bの参照との照合結果と未達の基準は[モデル別の検証](qwen3.8-27b.ja.md#検証)、速度とメモリは[主な数値](qwen3.8-27b.ja.md#主な数値)にあります。ほかのモデルは[モデル一覧](../README.ja.md#モデルとロードマップ)から各ガイドを参照してください。
