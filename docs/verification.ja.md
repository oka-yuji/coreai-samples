# 検証と計測

[トップ](../README.ja.md) · [English](verification.md) · [ランタイム](runtime.ja.md)

## 検証

Swiftの書式は `python3 Tools/Format/check_format.py --base origin/main` で検査します。
必要なツール、対象ファイルの選び方、回帰試験は[書式検査の手順](../Tools/Format/README.md)を参照してください。
検査処理はこのリポジトリ内で完結し、個人用の設定フォルダは必要ありません。

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

自前のランタイムの `chat` と `tools` は `--reload-interval` と `--reload-footprint-bytes` を受け取り、0 でそれぞれの条件を止めます。`--reload-graphics-growth-bytes` も受け取り、0 で graphics の増分の条件を止め、省くと資産ごとの既定のままです。`chat --new-conversation-every 10` は 10 プロンプトごとに新しいセッションを始めます。自前のランタイムのチャットの記録には `generated_tokens` があり、停止のトークンを除いて、サンプリングした出力の ID をすべて含みます(応答の上限に達したときの最後の未処理の ID も含む)。失敗した CLI のターンには、この項目がありません。Verification SPI は、依頼だけの検査を通って実行に入った最後の依頼を返します。デコードの前は空、成功すれば全体、デコードの失敗や取り消しではその時点の途中のサンプルです。依頼だけの検査で断られた依頼は実行のロックの前で止まり、前のスナップショット、処理済みの状態、会話の追跡を変えません。executor が動く前にフレームワークが断った場合も、スナップショットは変わりません。ターンとツールのステップの記録には、読み直しがあれば `reload_seconds` が入り、メモリの記録には graphics footprint のバイト数(graphics footprint の ledger と、その圧縮の ledger の和)が入ります。`chat --check-state` は、読み直した各ターンと最後の状態を確かめます。

```bash
COREAI_TEST_BUNDLES=/path/to/bundle swift test --filter ModelIntegrationTests
COREAI_TEST_BUNDLES=/path/to/bundle COREAI_TEST_RELOAD_INTERVAL=0 COREAI_TEST_RELOAD_FOOTPRINT_BYTES=1 swift test --filter ModelIntegrationTests.twoSessionsShareOneModel
```

閾値を低くした同時の依頼の実行では、少なくとも 1 回の読み直しが観測されなければなりません。CPU の方針の試験と偽のエンジンの試験に加えて、モデルを使う結果が要ります。

メモリの増え方を測るには、30 プロンプトの自前のランタイムの `chat` に `--memory-sample-turn 10 --memory-sample-turn 20 --memory-sample-turn 30` を足します。既定では off です。選んだターンでは、まず元のその場の `turn` の記録を出し、そのあと同じモデルを保ったまま 5 秒間推論せずに待ってから、ターンの番号、`wait_seconds`、メモリを取るまでの実際の `elapsed_seconds`、いつものメモリの項目を持つ別の `memory-sample` の記録を出します。状態の確認を頼んでいれば、そのあとに行い、最後の確認も含みます。重複したターンの番号や存在しないターンの番号は、読み込みの前に断ります。

5 秒の間隔は計測の決まりで、どの端末でもメモリが落ち着いた、回収が終わったことを保証するものではありません。その場の値とピークも、遅らせた標本と並べて残します。footprint の閾値の較正は、引き続きターン直後のその場の footprint の最大を使い、1.5 倍して 10 進の GB に切り上げます。観測の項目と待ちと取得の振る舞いの CPU だけの確認は、`swift test --package-path Tools/Verify` で流れます。

## 全位置の logits

`coreai-verify logits` は、自前のエンジンでトークン列を読み、開始位置から後の各位置の logits をファイルに書き出します。参照との照合や、塊の長さどうしの比較に使います。モデルを読むので、先に `AGENTS.md` にある重いジョブのロックを取ります。

```bash
cd Tools/Verify
swift build -c release
.build/release/coreai-verify logits --model <bundle> --tokens tokens.json --start 0 --chunk 4 --output logits.f16
```

| 指定 | 意味 |
|---|---|
| `--tokens` | トークン ID の JSON の配列(例: `[151644, 872, 198]`)。空の配列は断ります。各 ID は `language.vocab_size` 未満、トークンの数は `language.max_context_length` 以下にします。 |
| `--start` | logits を書き出す最初の位置で、0 から最後の位置まで。これより前のトークンは、エンジンの `callLengths(from:count:)` が返す呼び出しで読みます。1 回はエンジンの prefill の塊の長さ(記録の `prefix_chunk`)以下で、[呼び出しの形の規則](#呼び出しの形)が求めるところでは短くなります(最後の、または唯一の呼び出しも短いことがある)。その行は logits のファイルに書きません(`--prefill-output` のファイルには、その行が入ります)。`prefill` 関数があればそれで読み、logits の配列を作りません。ただし、呼び出しの形の規則が前置きを prefill の塊と違う分け方にするときは、`prefill` 関数が無いときと同じく `main` で読み、その logits は捨てます。その `main` の各呼び出しには、`--chunk` によらず、呼び出しの長さ × 語彙 × 2 バイトの logits の配列が要ります。`prefill` 関数が無いときは最大で前置きの塊の長さぶん(4,096 × 262,144 で 2 GiB)、Gemma 4 31B の資産で規則が前置きを分けたときは、最も長い呼び出しの 96 × 262,144 × 2 バイト = 48 MiB です。 |
| `--chunk` | `--start` から後を 1 回の `main` に通すトークンの数で、正の数。1 回ごとにちょうど 1 回の `main` の呼び出しで、最後の 1 回は短くなることがあります。 |
| `--output` | 出力のファイル。まだ無いファイルを指定します。あるファイルは上書きしません。 |
| `--main-output` | `NAME=FILE`。繰り返して指定できます。`main` の出力 `NAME` を、logits と同じ位置について `FILE` に書きます。形は `[位置数, 幅]` で、幅はその出力の幅です。`NAME` は `main` の logits 以外の出力で、形が `[1, 行数, 幅]` の float16 の配列でなければならず、そうでなければ `--start` から後の最初の `main` の呼び出しの前に失敗します。資産は、こうした出力の行の軸を動的な長さ、幅を固定の長さで宣言している必要があります。`FILE` はまだ無いファイルにします。 |
| `--prefill-output` | `NAME=FILE`。繰り返して指定できます。`prefill` 関数の出力 `NAME` を、`--start` より前の位置について `FILE` に書きます。形は `[start, 幅]` です。`--start` は 0 より大きく、その前のトークンを `prefill` 関数で読むこと(`prefix_function` が `prefill`)と、`NAME` が `prefill` 関数の同じ種類の出力であることが要り、そうでなければそのトークンを読む前に失敗します。`FILE` はまだ無いファイルにします。 |

位置 p の行は、トークン列の 0 番目から p 番目までを読んだ後の、次のトークンの分布です(`main` が位置 p について返した logits)。ファイルはヘッダの無い little-endian の float16 で、形は `[位置数, 語彙]`、位置数はトークンの数 − 開始位置です。各行はその `main` の呼び出しが返ったらすぐに書きます。メモリに要るのは 1 回の `main` の行(塊の長さ × 語彙 × 2 バイト)です。例えば語彙 262,144 で、塊 256 なら 128 MiB、16,384 位置を 1 つの塊にすると 8 GiB です。`--start` より前を `main` で読むとき(`prefill` 関数が無い資産と、Gemma 4 31B の資産で規則が前置きを分けるとき。後者は `--start` が 97〜112)は、その間も 1 回の呼び出しの長さ × 語彙 × 2 バイトが要り、`--chunk` を小さくしても減りません。`prefill` 関数が無い資産では最大で前置きの塊の長さぶん、規則で分けた前置きでは 96 × 262,144 × 2 バイト = 48 MiB です。ファイルは位置数 × 語彙 × 2 バイトです。失敗したときは自分で作ったファイルを消します。記録の行が出たファイルだけを完全なものとして扱います。コマンドが終わる前にプロセスが止まったときは、書きかけのファイルが残るので、同じ `--output` で流し直す前に消します。NumPy では `np.fromfile(path, dtype="<f2").reshape(shape)` で読めます。`--main-output` と `--prefill-output` のファイルも同じ形式で、失敗したときは logits のファイルと一緒に消します。コマンドが終わる前にプロセスが止まったときは、logits のファイルと同じく書きかけのまま残るので、同じ名前で流し直す前に消します。メモリでは、それぞれの出力が 1 回の呼び出しの行(呼び出しの長さ × 幅 × 2 バイト)を持ちます。

記録は `record` が `logits` の JSON の 1 行です。

| 鍵 | 内容 |
|---|---|
| `token_count`、`tokens_sha256` | トークンの数と、10 進の ID をコンマでつないだ文字列の SHA-256 |
| `start`、`chunk` | 指定の値 |
| `shape`、`dtype`、`byte_order` | `[位置数, 語彙]`、`float16`、`little` |
| `prefix_function`、`prefix_chunk`、`prefix_calls` | `start` より前のトークンを読んだ関数(`prefill`。`prefill` 関数が無いときと、呼び出しの形の規則が前置きを prefill の塊と違う分け方にするときは `main`。`start` が 0 のときは無い)、1 回の呼び出しのトークンの数の上限で、エンジンの prefill の塊の長さ(最後の、または唯一の呼び出しと、呼び出しの形の規則で短くした呼び出しは、それより短いことがある)、実際に呼んだ回数 |
| `main_calls` | `start` から後の `main` の呼び出しの回数 |
| `main_hash` | 資産の `main.hash` の 16 進(`main.mlirb` の SHA-256)。ファイルが無いときは無い |
| `requested_compute`、`expect_frequent_reshapes`、`cached`、`cache_query_error`、`specialize_s` | 自前のエンジンの特殊化の要求とキャッシュの状態。`own-load` の記録と同じ |
| `non_finite_count` | 書き出した logits の行の NaN と無限大の数。`--main-output` と `--prefill-output` のファイルは数えません。0 でなくてもコマンドは成功するので、この数を確かめます。 |
| `load_s`、`prefix_s`、`scoring_s` | エンジンの読み込み、`start` より前のトークンの読み込み、`start` から後の読み込みと書き出しの秒数 |
| `scoring_call_s` | `start` から後の `main` の呼び出しの中の秒数(行の読み出しとファイルの書き出しを含まない) |
| `outputs` | `--main-output` か `--prefill-output` を指定したときだけ。そのファイルごとの `function`(`main` か `prefill`)、`name`、`rows`、`width` と、中身の `sha256` |

このコマンドは `Engine` の Verification SPI を使います。`readInOneCall(_:usePrefillFunction:outputs:)` は、塊をちょうど 1 回の `main` の呼び出し(頼まれて、バンドルに `prefill` 関数があれば `prefill`)で読みます。`outputs` を渡さない呼び出しは、これまでどおり、`main` では logits だけを束縛し、`prefill` 関数では出力を束縛しません。`outputs` を渡すと、その関数のその出力も、呼び出しの全行について束縛し、`output(_:row:)` が、次の読み込み・`restore`・`reset`・`unload` まで、その 1 行を返します。関数が持たない名前、`main` の logits、形が `[1, 行数, 幅]` の float16 の配列でない出力は、呼び出しの前に `OutputRowError.unreadableOutput` を投げ、エンジンは変わりません。`output(_:row:)` は、直前の呼び出しが読まなかった出力、呼び出しの範囲外の行、幅の軸が連続でない配列で `OutputRowError` を投げます。`logits(row:)` は直前の `main` の呼び出しの 1 行を返し、最後の行は `logits()` と同じです。行を読めるのは、次の読み込み・`restore`・`reset`・`unload` までです。`prefill` だけで読んだ直後とこれらの後、直前の呼び出しの範囲外の行では `LogitRowError` を投げます。logits の配列の語彙の軸が連続でないとき(stride が 1 でない、または interleave の配置)も、別の値を読まずに `LogitRowError` を投げます。`prefillChunkSize` と、位置 `from` から `count` 個のトークンを読むときの呼び出しの長さの列を返す `callLengths(from:count:)` も、同じ SPI にあります。この読み込みは、`Engine.Stats` の `prefillTokens`・`prefillSeconds`・`decodeTokens`・`decodeSeconds` を変えません。

モデルの要らない確認は `swift test --package-path CoreAISamplesKit --filter 'LogitRowLayoutTests|OutputRowsTests'` と `swift test --package-path Tools/Verify -c release --filter PositionLogitsTests` です。バンドルを渡すと、モデルを使う試験が、最後の行を `logits()` と、1 トークンずつの塊を `step` と、2・4・8 の塊を 1 トークンずつの塊と比べます。前の 2 つはビットまで一致しなければなりません。長い塊は計算の順が変わるので、非有限値が無く、上位のトークンが違うのは 8 目盛りの同点近傍の規則が許す場合だけであることを求め、塊の長さごとに logits の最大絶対差を出力します。

```bash
COREAI_TEST_BUNDLES=/path/to/bundle swift test --package-path CoreAISamplesKit --filter ModelIntegrationTests.positionLogits
```

## 呼び出しの形

Gemma 4 31B の変換物には、Core AI が誤った logits を返す呼び出しの形が 1 つあります。8 トークン以上の呼び出しで、呼び出しの後の位置の総数が 97〜112 になるものです。#665 では、先頭 6 層の FP16 で logits が FP32 の参照から最大 6.19 ずれました(非有限値は出ていません)。`Engine` は、`metadata.json` の `source.hf_model_id` が `google/gemma-4-31B-it` の資産でだけ、この形を避けます。ほかの資産の呼び出しの長さは変えません。prefill の 2 つの経路(`prefill` 関数と `main`)はどちらも、prefill の塊の長さ以下で、そこで終わらない最も長い呼び出しを選びます。`prefill` 関数を持たない資産と、`coreai-verify logits` の前置きでは、位置 0 から、100 トークンを 96 + 4、112 トークンを 96 + 7 + 7 + 2 の呼び出しで読みます。`prefill` 関数を持つ資産では、`prefill` 関数が最後の 1 トークンを除いて読み(100 トークンは 96 + 3、112 トークンは 96 + 7 + 7 + 1)、`main` が最後の 1 トークンを読みます。規則は `Engine` が出どころのモデル ID で引く表で持ち、バンドルの項目にはしていません。`readInOneCall(_:usePrefillFunction:)` は渡された塊を今までどおり 1 回で読むので、誤りを再現できます。`coreai-verify logits` は、`--start` より前を `callLengths(from:count:)` で読み、`--start` から後は `--chunk` の長さのまま読みます。

Gemma 4 31B の `prefill` 関数を持つ資産では、`readInOneCall` で `prefill` 関数を 2 回以上続けて呼んだ直後の、複数トークンの `main` の呼び出しで、MPSGraph の assertion(`Failed to resolve dynamic dimensions for memref.alloc`)でプロセスが止まることがあります。エンジンを直接使った確認では、`prefill` 96 + 4 の後の `main` 4、96 + 7 の後の 7、50 + 3 の後の 3 で止まり、`prefill` 100 の後の `main` 4、96 + 4 の後の 8、96 + 5 の後の 4、96 + 7 + 7 + 2 の後の 2 では止まりませんでした。`prefill` 関数を持たない資産で同じ長さを読んだときも止まりませんでした。止まる条件の全体は分かっていません。`prefill` は最後に 1 トークンの `main` を呼び、下のモデルの確認では長さ 1〜300 のどの入力でも止まっていません。`coreai-verify logits` は、規則が前置きを prefill の塊と違う分け方にするとき(この資産では `--start` が 97〜112)、前置き全体を `main` で読み、その logits は捨てます。それ以外は今までどおり、`prefill` 関数があればそれで読みます。`prefill` 関数を持つ先頭 6 層の FP16 の資産で、128 トークンの入力に対し、`--start` 96〜113 のすべてと `--chunk` 1・2・4・7・8 の組(90 回)が止まらずに終わり、非有限値もありませんでした。書き出した logits と `--start 0 --chunk 1` の logits の差は、`--chunk` の呼び出しがこの形を含まない回ではどれも 0.754 以下でした。`--chunk 8` で `--start` が 96〜104 の回では、`--chunk` をそのまま読むので、8 トークンの書き出しの呼び出しが 97〜112 で終わります。その呼び出しの差は 1.34〜15.85、それらの回のほかの行は 0.512 以下でした。

モデルの要らない確認は `swift test --package-path CoreAISamplesKit --filter CallShapeTests` です。モデルを使う確認には、Gemma 4 31B のバンドルのフォルダを `:` 区切りで `COREAI_GEMMA4_SHAPE_BUNDLES` に、新しい JSON のファイルを `COREAI_GEMMA4_SHAPE_REPORT` に渡します。あるファイルと、出どころのモデル ID が違うバンドルは断ります。先に重いジョブのロックを取ります。`check.sh` は `COREAI_*` の変数を消すので、直接流します。

```bash
COREAI_GEMMA4_SHAPE_BUNDLES=/path/to/bundle-a:/path/to/bundle-b COREAI_GEMMA4_SHAPE_REPORT=/new/report.json swift test --package-path CoreAISamplesKit --filter ModelIntegrationTests.gemma4CallShapes
```

試験はバンドルごとに、固定の英語の文をバンドルの tokenizer で token にし、`<bos>` と続く 299 トークンを使います。`prefill` での読み込みと 1 回の呼び出しでの読み込みは別々のエンジンで行い、それぞれのエンジンで 1 トークンずつ読んだ logits を基準にします。2 関数の資産では、位置 0 から `prefill` 関数で 1 トークン、続けて 2 トークンを読んだエンジンが、次に位置 0 から `main` で 3 トークンを読むと、MPSGraph でプロセスが止まります(`Failed to resolve dynamic dimensions for memref.alloc`。2・3 トークンの後の 4 トークンも同じ)。新しいエンジンでは止まりません。長さ 1〜300 のそれぞれで、エンジンを空にしてから `prefill` で読んだ最後の位置、または 1 回の `main`(`readInOneCall`)で読んだ最後の位置を、そのエンジンの基準との絶対値の差の最大で比べます。`prefill` は全部の長さで 2.0 未満で、予定どおりの呼び出しをすること。1 回の `main` は 97〜112 のどれかで 2.0 以上になり、ほかの長さでは 2.0 未満であること。非有限値が無いこと。2.0 は、#665 で測った正常な形の差(0.17〜0.84)と誤りの形の差(3.06〜6.19)の間に置いた値です。記録には、バンドルごとに `main_hash`、規則、トークン、時間、2 つの基準が等しいかと、長さごとの `prefill` の呼び出しの長さの列、2 つの差、非有限値の数を書きます。

## 品質評価の出力の記録

`coreai-verify evaluate` は、品質評価の入力の token 列から、自前のエンジンで生成し、品質契約の出力の記録と実行の記録を書きます(例えば [Gemma 4 31B の品質契約](specs/gemma4-31b-quality-contract.md)の §5.1)。モデルを読むので、先に `AGENTS.md` にある重いジョブのロックを取ります。

```bash
cd Tools/Verify
swift build -c release
.build/release/coreai-verify evaluate --model <bundle> --inputs inputs.jsonl --contract ../LLMEvaluation/contract.json \
  --set primary --role fp16 --commit <commit> --output <run-folder>
```

| 指定 | 意味 |
|---|---|
| `--model` | バンドルのフォルダ。資産に `main.hash` が要ります。実行の記録で資産を示すのに使います。 |
| `--inputs` | 入力の記録。形は下に書きます。 |
| `--contract` | 契約の `contract.json`。その `version` と SHA-256 を実行の記録に書き、`generation.eos_token_id` を停止 token にします。`--role` は、その `roles` の鍵のどれかにします。 |
| `--set`、`--role` | 実行の記録に書く集合の名前(例: `primary`)と役割(例: `fp16`)。 |
| `--commit` | 実行の記録に書くリポジトリの commit。コマンドは git を呼びません。 |
| `--output` | 既にあるフォルダの中の新しい実行のフォルダか、再開する同じ実行のフォルダ。`run.json` の無い既存のフォルダは、空か、空の `outputs.jsonl` だけ(0 行の `progress.json` があってもよい)のとき(`run.json` を書く前に止まった開始が残す形)に限り、新しい実行として始めます。それ以外の `run.json` の無いフォルダは断ります。 |
| `--max-minutes` | 任意。コマンドを始めてからこの分数が過ぎたら、いまの item を書いたところで正常に終わります。そのため、残りの item があれば 1 件は必ず流れます。 |
| `--limit` | 任意。入力の記録の先頭から N 件だけ流します(確認用)。 |

`inputs.jsonl` は 1 行に 1 件の入力の記録で、JSON のオブジェクトに `id`(文字列)、`input_ids`(token ID の配列)、`input_sha256`(ID を 10 進で書いてコンマでつないだ UTF-8 の SHA-256。`logits` の `tokens_sha256` と同じ)、`max_new_tokens`(その item が生成してよい token の数の上限。停止 token も数える)を持ちます。ほかの項目は読みません。モデルを読む前に全件を確かめ、同じ `id` が 2 回ある、`input_ids` が空か `language.vocab_size` の外の ID を含む、`max_new_tokens` が正でない、`input_sha256` が ID と合わない、のどれかがあれば、何も書かずに止まります。

item ごとに、エンジンを空にし、入力を `prefill` で読み(バンドルに `prefill` 関数があればそれで読む)、各 step で最後の位置の logits(float16、語彙の全体)を読みます。logits に非有限値がある step で、その item を終えます。そうでなければ、logits の大きい順に 2 つの token を `top2` に足し(同じ値なら小さい ID が先)、その確率は、最大の logit を引いてから倍精度で取った語彙の全体の温度 1 の softmax の値にします。順位は logits で決めるので、1 位から大きく離れた 2 位は、確率が 0 に丸まっても 2 位のままです。その 1 位が生成する token です。停止 token なら、`stop` を `stop_token` にして item を終えます。それ以外は `output_ids` に足し、`max_new_tokens` に達したら `stop` を `limit` にして終え、達していなければ `step` でその token を読みます。停止 token と、上限に達した最後の token は読みません。

実行のフォルダには 3 つのファイルを置きます。

| ファイル | 内容 |
|---|---|
| `outputs.jsonl` | item ごとの出力の記録。1 行に 1 件の JSON のオブジェクトで、item を終えるごとに追記してディスクへ同期します |
| `progress.json` | `outputs.jsonl` の確定した行の数 `lines` と、それらの行の改行を含むバイト列の SHA-256 `sha256`、これから書く 1 行を足した後の SHA-256 `next_sha256`(行を書いている間だけある)、実行の `id` と再開で照合する条件を、鍵の順を固定した JSON にしたものの SHA-256 `run`。行を追記して同期する前に、atomic に置き換えてその行を予告し、次の予告か実行の終わりで、その行を確定させます。これと、これが指す `outputs.jsonl` の行が、済んだ item の正本です |
| `run.json` | 実行の記録。実行の始めと終わりに書きます |

| 出力の記録の項目 | 値 |
|---|---|
| `id`、`input_sha256` | 入力の記録の値 |
| `input_tokens` | 入力の token の数 |
| `output_ids` | 最初の停止 token より前に生成した ID |
| `stop`、`stop_token` | `stop_token`(`stop_token` にその停止 token の ID)か、`limit`(`stop_token` は無い) |
| `top2` | 生成した token ごとに(停止 token も数える)1 件の `[1 位の ID, 1 位の確率, 2 位の ID, 2 位の確率]`。`limit` で終わった item の件数は `output_ids` と同じ |
| `error` | 誤りなく終わった item には無い。それ以外は終わった理由: logits の非有限値か、`prefill`・`step` が投げた誤り(文脈の上限など)。その item は、それまでの `output_ids` と `top2` を残して `stop` を `limit` にし、実行は次の item へ進みます。 |
| `non_finite` | item を終えた step の logits の非有限値の数。それ以外は 0 |

| 実行の記録の項目 | 値 |
|---|---|
| `id` | 実行を最初に始めたときに作る UUID。再開しても変えない。`progress.json` はこれに結び付く |
| `contract` | `contract.json` の `version` と `sha256` |
| `set`、`role` | 指定の値 |
| `asset` | バンドルの `name`、`main_hash`、`metadata.json` に `source.hf_model_id` があれば `source_model_id` |
| `implementation` | `command`(`coreai-verify evaluate`)、`commit`、`os_version`、`os_build` |
| `call_shape` | エンジンの `prefill_chunk`、`prefill_function`、`expect_frequent_reshapes`、`requested_compute` と、`metadata.json` の `language.max_context_length` で決まる実効の文脈の上限 `context_limit` |
| `inputs` | 入力のファイルの `sha256` と `count` |
| `items` | `run.json` を書いたとき(各回の始めと、`--max-minutes`・`--limit` での終わりを含む正常な終わり)の、済んだ item の数。実行の途中と、異常に止まった後は、これより多く済んでいることがあります。済んだ item の正本は `progress.json` と、それが指す `outputs.jsonl` の行です。 |
| `started`、`finished` | 実行を最初に始めた時刻と、最後に終えた時刻。実行の途中と、誤りで止まった後は `finished` が無い |
| `load` | 最初の回のエンジンの読み込みの `cached`、`specialize_s`、`load_s` |

`--output` に `run.json` があるときは、同じ実行のものでなければなりません。`contract`、`set`、`role`、`asset.main_hash`、`asset.source_model_id`、`inputs`、`call_shape`、`implementation` が今回と同じでなければ、生成する前に止まります。エンジンは `asset.source_model_id` で入力の読み方を選ぶので、これも、無いことを含めて比べます。`id` の無い `run.json`(このコマンドの前の版が書いたもの)は再開できません。`progress.json` があり、その `run` が `run.json` の `id` と条件の SHA-256 と同じでなければなりません。`run.json` の `items` は 0 から入力の記録の件数まで、`progress.json` の `lines` は 0 から `outputs.jsonl` の改行で終わる行の数まででなければなりません。`outputs.jsonl` の最後の改行より後ろのバイト(書いている途中で切れた行)は脇に置きます。改行で終わる行が `items` 行以上あり、かつ、`lines` 行でその SHA-256 が `sha256` と同じか、1 行多くて `next_sha256` と同じ(予告した行を書き終えた後に止まった)なら、それらの行を済んだ item とし、切れたバイトを消して、その item を流し直します。どちらでもなければ、何も生成せず、何も消さずに止まります。改行で終わる行は、どの場合も消しません。済んだ item は流し直しません(誤りで終わった item も)。その中に、入力の記録に無い `id` か、2 回出てくる `id` があれば止まります。この検査が見つけるのは、途中で切れた末尾、`outputs.jsonl` と `progress.json` の食い違い(片方だけが古い・新しい・変わった)、条件の違う実行のファイル、条件が同じでも別の実行のファイル、`run.json` の `items` より少ない行数です。見つけないのは、同じ回の途中の時点へ `outputs.jsonl` と `progress.json` を揃えて戻すこと(その時点での中断と区別できないので、戻した時点から続けます)と、`outputs.jsonl` と `progress.json` を互いに合うように意図して書き換えること(`run.json` を変えなくても通ります。再開は行の意味を検査し直しません。行の意味は生成の試験が受け持ちます)です。再開した実行は `started` と `load` を残し、`items` と `finished` を更新します。

標準出力には item ごとに 1 行の JSON を出します。`record` が `evaluate` で、`id`、`input_tokens`、`output_tokens`、`stop`(`stop_token`、`limit`、item に誤りがあれば `error`)、`seconds` を持ちます。文は出しません。取り消しと、入力の記録・契約・バンドル・実行のフォルダを読めないときは、item の誤りにせず、コマンドを止めます。

モデルの要らない確認は `swift test --package-path Tools/Verify -c release --filter "EvaluationRecordTests|EvaluationRunTests"` です。

## 状態を持たない関数

`coreai-verify function` は、状態を持たない Core AI の資産の関数を、ファイルで渡した入力で 1 つ動かし、出力をファイルと記録に書きます。Gemma 4 31B の変換の MTP の drafter のような、言語モデルでない資産を、アプリと同じように Core AI で動かすためのものです。モデルを読むので、先に `AGENTS.md` にある重いジョブのロックを取ります。

```bash
cd Tools/Verify
swift build -c release
.build/release/coreai-verify function --model drafter.aimodel --function draft_7 --inputs inputs/inputs.json --output out --repeat 3
```

この例は、`drafter.aimodel`([合成の drafter の書き出し](../Tools/Conversion/recipes/gemma4_31b/README.md#export-of-a-synthetic-drafter)を `--steps 7` で流して得る `unrolled.aimodel` か `both.aimodel`。どちらにも関数 `draft_7` があります)と、`inputs.json` とそれが指すデータのファイルを入れた `inputs` のフォルダがあるフォルダで流します。

| 指定 | 意味 |
|---|---|
| `--model` | 資産の `.aimodel` のフォルダ。バンドルのフォルダではありません。 |
| `--function` | 関数の名前。資産は複数の関数を持てます。 |
| `--inputs` | 入力を並べる JSON のファイル(下に書きます)。 |
| `--output` | 新しいフォルダ。在るパスは、リンクも含めて断ります。 |
| `--repeat` | 同じ入力で呼ぶ回数。1 以上で、既定は 1。どの呼び出しの出力も、最初の呼び出しの出力と byte で同じでなければなりません。 |

入力の一覧は、鍵がちょうど `inputs` だけの JSON のオブジェクトで、値は入力ごとに 1 つのオブジェクトの配列です。各オブジェクトの鍵は、ちょうど `name`・`dtype`・`shape`・`file` です:

```json
{"inputs": [
  {"name": "input_ids", "dtype": "int32", "shape": [1, 1], "file": "input_ids.bin"},
  {"name": "hidden_states", "dtype": "float16", "shape": [1, 1, 48], "file": "hidden_states.bin"},
  {"name": "position_ids", "dtype": "int32", "shape": [1, 1], "file": "position_ids.bin"},
  {"name": "k_sliding", "dtype": "float16", "shape": [1, 1, 2, 5, 16], "file": "k_sliding.bin"},
  {"name": "v_sliding", "dtype": "float16", "shape": [1, 1, 2, 5, 16], "file": "v_sliding.bin"},
  {"name": "k_full", "dtype": "float16", "shape": [1, 1, 1, 9, 32], "file": "k_full.bin"},
  {"name": "v_full", "dtype": "float16", "shape": [1, 1, 1, 9, 32], "file": "v_full.bin"}
]}
```

これは `draft_7` の 7 つの入力の全部で、sliding window の key と value が 5 位置、full attention のものが 9 位置の場合です。ファイルの大きさは 4、96、4、320、320、576、576 バイトです。

`dtype` は `int32`・`float16`・`float32` のどれかです。`shape` は、1 以上の整数の、空でない配列で、整数の書き方で書きます。`1.0` と `1e0`、0、負の数、真偽値、文字列は断ります。`file` は、入力の一覧と同じフォルダの中のファイルの名前です。空でなく、`/` と NUL 文字を含まず、`.` と `..` でないものに限ります。ファイルの中身は、要素を little-endian の値で、行優先の順に、ヘッダなしで並べたもので、大きさは形の積に型の幅(`float16` は 2 バイト、`int32` と `float32` は 4 バイト)を掛けた値とちょうど同じです。NumPy の dtype は、`int32` が `"<i4"`、`float16` が `"<f2"`、`float32` が `"<f4"` で、`array.astype("<f2").tofile(path)` で書け、`np.fromfile(path, dtype="<f2").reshape(shape)` で読めます。同じ名前のメンバーが 1 つのオブジェクトに 2 回現れる場合は、検出しません。

次の場合は、特殊化の前に、どれかを言って止まり、出力のフォルダを作りません:

1. 出力先のパスが在る(リンクも含みます)。
2. 入力の一覧が通常のファイルでない(リンク・フォルダ・名前付きパイプは通常のファイルではありません)、または中身が上の形でない。
3. 資産にその名前の関数が無い、または関数が state を持つ。この口は、状態を持たない関数だけを動かします。
4. 一覧の名前が、関数の入力の名前とちょうど同じでない(どの名前も 1 回だけ並べます)。
5. 入力の型(dtype)または次元の数が資産の記述と違う、または固定の次元が違う。資産が `?` と書く動的な次元は、1 以上のどんな大きさも通します。
6. データのファイルが通常のファイルでない(リンク・フォルダ・名前付きパイプを断ります)、または大きさが形と型から決まる値と違う。ファイルは、待たずに、リンクを辿らずに開き、種類と大きさを確かめてから読みます。

入力の型は、資産の要約(特殊化しないで読めます)にある `NDArray (<型>, <次元>)` の文字列から、`Int32`・`Float16`・`Float32` のものだけを読みます。資産がこれ以外の形で書く入力は断ります。

検査が全部通ったら、出力のフォルダを作り(作れないとき、例えば親のフォルダが無いときは、モデルを読む前に、理由を言って止まります)、自前のエンジンと同じ設定(計算ユニットの自動選択、`expectFrequentReshapes` は真。変える指定はありません)で資産を特殊化し、関数を `--repeat` 回呼びます。出力のフォルダの中身は次のとおりです:

| ファイル | 内容 |
|---|---|
| `<出力の名前>.bin` | 最後の呼び出しの出力ごとに 1 つ(関数の出力の順)。要素を little-endian の値で、行優先の順に、ヘッダなしで並べたもので、型は記録がその出力について書く `dtype` です(NumPy の dtype は上のとおり)。ファイル名にできない出力の名前(空、`/` か NUL 文字を含む、ファイル名が 255 バイトを超える)は、呼び出しの後、ファイルを書く前に断ります。 |
| `function.json` | 下の記録。標準出力に出す行と同じものです。 |
| `failure.json` | 最初に書き、実行が成功したときに消します。在れば、その実行は成功していません(下)。 |

記録は `record` が `function` の JSON の 1 行です:

| 鍵 | 内容 |
|---|---|
| `byte_order` | `little` |
| `main_hash` | 資産の `main.hash` の 16 進(`main.mlirb` の SHA-256)。ファイルが無いとき、通常のファイルでないとき(リンク・フォルダ・名前付きパイプ)は無い |
| `function` | 関数の名前 |
| `inputs` | 入力ごとに 1 つ(入力の一覧の順): `name`・`dtype`・`shape`・`bytes`(形の積に型の幅を掛けた値で、ファイルの大きさ)・ファイルの中身の `sha256` |
| `outputs` | 出力ごとに 1 つ: 同じ鍵と、出力のフォルダの中のファイル名 `file` |
| `requested_compute`、`expect_frequent_reshapes`、`cached`、`cache_query_error`、`specialize_s`、`load_s` | 特殊化の要求とキャッシュの状態。`own-load` の記録と同じ。`load_s` はキャッシュの照会から関数が読み込まれるまでの秒数。`cache_query_error` は誤りの説明で、ローカルのパスを含むことがあるので、これがある生の記録をそのまま公開しない |
| `call_s` | 呼び出しごとの秒数(順) |
| `repeat_identical`、`repeat_differing_calls` | どの呼び出しも最初の呼び出しと同じ出力を返したか。違った呼び出しの番号(2 から)。名前・型・形・byte のどれかが違えば違いです |

出力のフォルダに最初にすることは、`failure.json` を書くことです。書けないときは、作ったフォルダを消します(消せなければ、そう言います)。そして理由を言って、モデルを読む前に止まります。中身は `{"kind": ..., "record": "function-failure"}` の 1 行だけです。`kind` は最初は `incomplete` で、実行の終わり方が分かった時点で、それを表す `kind` に書き直します。書き直せなかったときは `incomplete` のまま残り、process が止められるなど、跡を残さずに止まったときも `incomplete` が残ります。

| `kind` | 実行の終わり方 |
|---|---|
| `incomplete` | 上のとおり、書き直されなかった。 |
| `load` | 特殊化、または関数の読み込みが失敗した。 |
| `call` | 呼び出しが失敗した。 |
| `write` | 出力か `function.json` の書き込み、または `failure.json` の除去が失敗した(断った出力の名前を含む)。 |
| `cancelled` | task が取り消された。関数を読む前、各呼び出しの前、出力を書く前に確かめます。 |
| `repeat_mismatch` | 最初の呼び出しと違う出力を返した呼び出しがあった。最後の呼び出しのファイルと、`repeat_identical` が偽の `function.json` を書き、記録を出して、終了コード 1 で終わります。 |

`load`・`call`・`write`・`cancelled` は、その原因の誤りで終わります。成功する実行は、`function.json` を書いてから `failure.json` を消し、消せないときは実行が失敗します。決まりは 1 つです: 出力のフォルダに `failure.json` が在れば、その実行は成功していません。

モデルの要らない確認は `swift test --package-path Tools/Verify -c release --filter "Function|InputFile|RunFunction|AIModelFunctionSource"` です。入力の一覧、関数の記述との照合、データのファイル、記録・失敗の記録・取消での停止を、モデルの代わりの偽の実装で確かめ、テンソルの変換を SDK の `NDArray` で確かめます。資産での実行は別の確認です。

## 計測項目と境界

依頼ごとの計測は、明示したときだけ行います。`chat --runtime own --measure summary` は各 `turn` の記録に `measurement` を足し、`--measure trace` はさらにサンプルごとの行を足します。既定の `--measure off` では、この項目は無く、生成、停止、エラー、メタデータ、使用量は、この指定が無いときと同じです。`tools` は `--measure` を断り、`bench` にはこの指定がありません。値は `CoreAIBackend` が採り、`coreai-verify` が出力します。Verification SPI の `lastRequestMeasurement()` は、`lastGeneratedTokens()` と同じく実行に入った最後の依頼の計測を返し、その依頼が生成の前に終わったとき(文脈あふれや読み込みの失敗など)は何も返しません。`turn` の記録は、そのターンの依頼に計測が無ければ `measurement` を持ちません。実行の前に断られた依頼では SPI に前の依頼の計測が残るので、CLI は `request_id` を前のターンのものと比べます。時間は 1 つの単調な時計で測った秒、回数は整数です。

ランタイムの生成は、依頼の検査、テンプレートの描画、文脈の詰め方、読み直しの後に始まります。読み直しの時間は別に `reload_seconds` に入ります。読み込みの後の段階は切れ目なく続くので、完了した依頼では、丸めの誤差を除いて `read_seconds` と 5 つの段階の合計の和が `total_seconds` に一致します。エラーで終わった依頼では、その時に進んでいた段階の時間は `total_seconds` だけに入ります。

| 項目 | 開始 → 終了 | 数える対象 | 単位 |
|---|---|---|---|
| `Engine.Stats.decodeSeconds` | 各 `step`: forward の開始(入力の準備を含む)→ `function.run` の後の forward の終了 | 完了した step(`decodeTokens`) | s |
| `Engine.Stats.prepareSeconds` | prefill の各チャンクと各 step の forward: 開始 → `function.run` の直前。`measuresPreparation` が true の間だけ加算します。ランタイムは計測する依頼でこれを on にするので、計測しない依頼ではここで時計を読みません | forward | s |
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
| speed の `specialize_s`、`cached` | own の `Engine.LoadStats` の特殊化の時間。キャッシュ済みのモデルでは 0。cache miss のときは、特殊化の前の資産の幅の検査(要約の読み取り)の時間も含む。以前の記録は含まないので、過去の値と比べるときは注意する | — | s |
| speed の `cache_query_error` | own の `Engine.LoadStats.cacheQueryError`。特殊化キャッシュの照会が誤りを投げ、続く特殊化が成功したロードで、照会の誤りの説明文が入る。照会が項目を見つけた(hit)ロード、誤りなく見つけなかった(miss)ロードと、official の行には無い。1 回の `speed-run`(1 プロセスで、ロードは 1 回)が `--repeat` で出す行は、どれも同じ値。`speed` が起動する子プロセス(round と engine ごと)は別々にロードするので、子の間では変わりうる。チャットの `load` の記録と `own-load` の記録にも同じ鍵が出る。説明文は誤りの `String(describing:)` で、手元のパスを含みうるので、生の記録をそのまま公開しない。[共通の記録の規約](benchmarking.ja.md)の手順 7 のとおりに除いてから公開する | — | — |
| speed の `max_tokens`、`generated_tokens` | 要求した上限(以前の `decode_tokens`)と、`onToken` または公式の出力列で実際に受け取ったトークンの数 | トークン | トークン |
| speed の `prefill_s`、`decode_s` | 実行の開始 → 最初のトークン、最初のトークン → 最後のトークン。最初のトークンを観測しなかった回では `prefill_s` と `prefill_tok_s` は無く(欠測)、`decode_s` は 0 | — | s |
| speed の `decode_tok_s` | `(generated_tokens − 1) / decode_s`。2 トークン未満では無い | トークン | トークン/s |

回数は、`prompt_tokens`、`prefilled_tokens`(この依頼でエンジンが読んだトークン。終わったチャンクごとに数えるので、読み込みが途中で失敗しても、それまでに終わったチャンクの分は残ります。プロンプトをすべて使い回したときや、最初のチャンクの前に失敗したときは 0)、`sampled_tokens`(`generated_tokens` と同じ並び。停止のトークンを含まず、上限に達したときの最後の未処理のトークンを含む)、`steps`、`sends`(1 件以上のイベントを送った送信。drain を含む)です。`outcome` は `stopToken`、`maxTokens`、`cancelled`、`failed` のどれかで、後の 2 つには `error` が付きます。speed の記録の `stop_reason` は、自前のエンジンでは件数の規則で決めます。ここでは停止のトークンを無視するので上限で終わり、`maxTokens` になります(上限より前に終わったときは `stopToken`)。公式のエンジンでは iterator が終わった理由(`InferenceOutputSequence.stopReason`)で、`maxTokens`、`stopToken`(EOS)、`stopSequence`、`cancelled`、`error`、`unknown` のどれかです。sequential は残りの文脈に合わせて要求数を縮めるため、`max_tokens` 未満でも `maxTokens` になり得ます。`trace` の行は `sampler.next` ごとに 1 行あるので、`stopToken` の依頼では `sampled_tokens + 1` 行になります。停止のトークンの行には `sample_seconds` だけがあり、後に step が無かった行には `step_seconds` がありません。

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
| 実際の演算の配置 | 返す公開 API がありません。`observed_compute` は常に `unknown` です。own のロードと speed の記録の `requested_compute` と `expect_frequent_reshapes` は、特殊化で要求した設定にすぎません。自前のランタイムは計算ユニットの自動選択を要求するので、`requested_compute` は `default`、`expect_frequent_reshapes` は `true` のままです。 |
| 公式の特殊化の時間とキャッシュの命中 | `EngineFactory` からは観測できません。official の speed の記録には `specialize_s`、`cached`、`requested_compute` がありません。キャッシュの状態は手順で固定します。 |
| `peak_footprint_bytes` | 測っていません。`footprint_bytes` は標本で、ピークではありません。 |
| `os_proc_available_memory` | macOS の SDK では使えないため、記録しません。 |

どの `memory` にも、`status`(`ok` か `failed`)、`captured_at`(ISO 8601 の wall clock)、`unit`(`B`)があります。`error`(例: `task_info returned 5`)は失敗したときだけ、`rss_bytes`、`footprint_bytes`、`graphics_footprint_bytes` は成功したときだけ出ます。speed の記録は、`getrusage` が失敗すると `peak_rss_bytes` を出しません。まとめの記録は、成功した取得の最大とその件数 `peak_rss_runs`、速度のある実行の中央値とその件数 `decode_runs` を出します。欠けた値を 0 として数えません。

`summary` は合計と回数を採ります。計測はサンプルごとに最大 6 回(ループで 4 回、step の forward で 2 回)時計を読み、`trace` はさらに行を保存します。結果を比べる前に、同じコマンドを `--measure off` と `--measure trace` で交互に流す A/A の確認をし、`turn` の記録の `total_s` と `decode_tok_s` を比べます。

```bash
.build/release/coreai-verify chat --model <bundle> --runtime own --reasoning off --measure trace --max-tokens 32 --prompt "List three fruits, one per line."
```

## モデル別の結果

Qwen3.8 27Bの参照との照合結果と未達の基準は[モデル別の検証](qwen3.8-27b.ja.md#検証)、速度とメモリは[主な数値](qwen3.8-27b.ja.md#主な数値)にあります。ほかのモデルは[モデル一覧](../README.ja.md#モデルとロードマップ)から各ガイドを参照してください。
