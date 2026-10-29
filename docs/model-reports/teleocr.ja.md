# TeleOCR

[トップ](../../README.ja.md) · [English](teleocr.md)

文書OCR

<!-- model-report:start -->
<a id="model-report"></a>

## 共通の検証サマリー

[計測プロトコル](../benchmarking.ja.md) · [元の記録](records/teleocr.json)

元の測定日・入力条件を保持しています。行ごとに異なるrunを使う場合があり、条件をそろえた速度比較ではありません。旧記録で不足する条件も明示します。未計測はゼロや合格を意味しません。

### デモ対応

| OS | 状態 | 確認した範囲・残る確認 | 根拠 |
|---|---|---|---|
| macos | — | このOSのネイティブデモは未提供です。 | — |
| ios | — | このOSのネイティブデモは未提供です。 | — |

### 性能と品質

| 指標 | 結果 | 集計・件数 | 条件と根拠 |
|---|---|---|---|
| バンドル・資産容量 | 2,847,208,089 B | max; n=20 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=prepared。評価したbundleの論理file bytes。 |
| バンドル・資産容量 | 2,420,910,034 B | max; n=20 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=prepared。評価したbundleの論理file bytes。 |
| バンドル・資産容量 | 2,200,708,898 B | max; n=20 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=prepared。評価したbundleの論理file bytes。 |
| バンドル・資産容量 | 2,186,926,997 B | max; n=20 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=prepared。評価したbundleの論理file bytes。 |
| 特殊化キャッシュ容量 | 6,686,155,073 B | max; n=20 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=prepared。F4後の重複しないasset-hash cache directoryの論理bytes。 |
| 特殊化キャッシュ容量 | 4,498,192,946 B | max; n=20 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=prepared。F4後の重複しないasset-hash cache directoryの論理bytes。 |
| 特殊化キャッシュ容量 | 4,277,991,298 B | max; n=20 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=prepared。F4後の重複しないasset-hash cache directoryの論理bytes。 |
| 特殊化キャッシュ容量 | 4,702,849,136 B | max; n=20 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=prepared。F4後の重複しないasset-hash cache directoryの論理bytes。 |
| 初回ロード（特殊化を含む） | 42.50915 s | single; n=1 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=cold。対象特殊化entryを削除。OS file/Metal cacheはflushしていない。 |
| 初回ロード（特殊化を含む） | 42.5374818 s | single; n=1 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=cold。対象特殊化entryを削除。OS file/Metal cacheはflushしていない。 |
| 初回ロード（特殊化を含む） | 42.8337595 s | single; n=1 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=cold。対象特殊化entryを削除。OS file/Metal cacheはflushしていない。 |
| 初回ロード（特殊化を含む） | 41.5460711 s | single; n=1 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=cold。対象特殊化entryを削除。OS file/Metal cacheはflushしていない。 |
| 特殊化済みキャッシュからのロード | 1.25549085 s | median; n=20 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=prepared。新process・特殊化済みcache。推論時間を含まない。 |
| 特殊化済みキャッシュからのロード | 1.37302458 s | p95; n=20 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=prepared。新process・特殊化済みcache。推論時間を含まない。 |
| 特殊化済みキャッシュからのロード | 0.968584083 s | median; n=20 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=prepared。新process・特殊化済みcache。推論時間を含まない。 |
| 特殊化済みキャッシュからのロード | 1.2408105 s | p95; n=20 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=prepared。新process・特殊化済みcache。推論時間を含まない。 |
| 特殊化済みキャッシュからのロード | 0.948625354 s | median; n=20 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=prepared。新process・特殊化済みcache。推論時間を含まない。 |
| 特殊化済みキャッシュからのロード | 1.10593583 s | p95; n=20 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=prepared。新process・特殊化済みcache。推論時間を含まない。 |
| 特殊化済みキャッシュからのロード | 1.15482544 s | median; n=20 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=prepared。新process・特殊化済みcache。推論時間を含まない。 |
| 特殊化済みキャッシュからのロード | 1.32300721 s | p95; n=20 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=prepared。新process・特殊化済みcache。推論時間を含まない。 |
| 要求・パイプライン完了時間 | 6.1443145 s | median; n=20 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=prepared。画像復号・page推論・表/Markdown組立。loadとfile I/Oは除外。 |
| 要求・パイプライン完了時間 | 6.78078596 s | p95; n=20 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=prepared。画像復号・page推論・表/Markdown組立。loadとfile I/Oは除外。 |
| 要求・パイプライン完了時間 | 6.098321 s | median; n=20 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=prepared。画像復号・page推論・表/Markdown組立。loadとfile I/Oは除外。 |
| 要求・パイプライン完了時間 | 6.55792796 s | p95; n=20 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=prepared。画像復号・page推論・表/Markdown組立。loadとfile I/Oは除外。 |
| 要求・パイプライン完了時間 | 6.00703044 s | median; n=20 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=prepared。画像復号・page推論・表/Markdown組立。loadとfile I/Oは除外。 |
| 要求・パイプライン完了時間 | 6.49287892 s | p95; n=20 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=prepared。画像復号・page推論・表/Markdown組立。loadとfile I/Oは除外。 |
| 要求・パイプライン完了時間 | 8.58823983 s | median; n=20 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=prepared。画像復号・page推論・表/Markdown組立。loadとfile I/Oは除外。 |
| 要求・パイプライン完了時間 | 10.652187 s | p95; n=20 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=prepared。画像復号・page推論・表/Markdown組立。loadとfile I/Oは除外。 |
| ピークRSS | 10,058,629,120 B | single; n=1 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=cold。loadを含むprocess生涯RSS peak。footprintと合算しない。 |
| ピークRSS | 9,076,244,480 B | max; n=20 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=prepared。loadを含むprocess生涯RSS peak。footprintと合算しない。 |
| ピークRSS | 6,723,731,456 B | single; n=1 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=cold。loadを含むprocess生涯RSS peak。footprintと合算しない。 |
| ピークRSS | 5,766,676,480 B | max; n=20 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=prepared。loadを含むprocess生涯RSS peak。footprintと合算しない。 |
| ピークRSS | 6,349,389,824 B | single; n=1 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=cold。loadを含むprocess生涯RSS peak。footprintと合算しない。 |
| ピークRSS | 5,349,982,208 B | max; n=20 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=prepared。loadを含むprocess生涯RSS peak。footprintと合算しない。 |
| ピークRSS | 8,231,796,736 B | single; n=1 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=cold。loadを含むprocess生涯RSS peak。footprintと合算しない。 |
| ピークRSS | 7,093,469,184 B | max; n=20 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=prepared。loadを含むprocess生涯RSS peak。footprintと合算しない。 |
| ピークphysical footprint | 3,781,090,096 B | single; n=1 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=cold。loadを含むprocess生涯footprint peak。RSSと合算しない。 |
| ピークphysical footprint | 3,276,495,136 B | max; n=20 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=prepared。loadを含むprocess生涯footprint peak。RSSと合算しない。 |
| ピークphysical footprint | 3,825,079,552 B | single; n=1 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=cold。loadを含むprocess生涯footprint peak。RSSと合算しない。 |
| ピークphysical footprint | 3,337,425,624 B | max; n=20 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=prepared。loadを含むprocess生涯footprint peak。RSSと合算しない。 |
| ピークphysical footprint | 4,019,606,736 B | single; n=1 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=cold。loadを含むprocess生涯footprint peak。RSSと合算しない。 |
| ピークphysical footprint | 3,356,381,744 B | max; n=20 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=prepared。loadを含むprocess生涯footprint peak。RSSと合算しない。 |
| ピークphysical footprint | 3,875,493,240 B | single; n=1 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=cold。loadを含むprocess生涯footprint peak。RSSと合算しない。 |
| ピークphysical footprint | 3,356,791,632 B | max; n=20 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=prepared。loadを含むprocess生涯footprint peak。RSSと合算しない。 |
| 推論中の常駐メモリ | 未計測 | — | 推論中の常駐量は未採取。要求後のrawメモリで代用しない。 |
| 解放後メモリの基準との差 | 未計測 | — | 同一process内の解放後と基準の差は未計測。 |
| 課題達成率 | 0.942857143 ratio | aggregate; n=35 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・synthetic。良判定かつEOS完了の件数/全予定入力。条件付き・欠測は成功の分子へ含めない。 |
| 課題達成率 | 0.75 ratio | aggregate; n=24 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・omnidocbench。良判定かつEOS完了の件数/全予定入力。条件付き・欠測は成功の分子へ含めない。 |
| 課題達成率 | 0.942857143 ratio | aggregate; n=35 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・synthetic。良判定かつEOS完了の件数/全予定入力。条件付き・欠測は成功の分子へ含めない。 |
| 課題達成率 | 0.75 ratio | aggregate; n=24 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・omnidocbench。良判定かつEOS完了の件数/全予定入力。条件付き・欠測は成功の分子へ含めない。 |
| 課題達成率 | 0.942857143 ratio | aggregate; n=35 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・synthetic。良判定かつEOS完了の件数/全予定入力。条件付き・欠測は成功の分子へ含めない。 |
| 課題達成率 | 0.75 ratio | aggregate; n=24 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・omnidocbench。良判定かつEOS完了の件数/全予定入力。条件付き・欠測は成功の分子へ含めない。 |
| 課題達成率 | 0.942857143 ratio | aggregate; n=35 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・synthetic。良判定かつEOS完了の件数/全予定入力。条件付き・欠測は成功の分子へ含めない。 |
| 課題達成率 | 0.75 ratio | aggregate; n=24 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・omnidocbench。良判定かつEOS完了の件数/全予定入力。条件付き・欠測は成功の分子へ含めない。 |
| 参照との誤差（定義は根拠参照） | 0.0571428571 error | aggregate; n=35 | [E3](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・synthetic。不一致または非EOSのページ対/結果がある対。欠測は別計上。logitのnRMSEではない。本文差0件、本文一致で他の予測差2件。 |
| 参照との誤差（定義は根拠参照） | 0.0416666667 error | aggregate; n=24 | [E3](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・omnidocbench。不一致または非EOSのページ対/結果がある対。欠測は別計上。logitのnRMSEではない。本文差1件、本文一致で他の予測差0件。 |
| 参照との誤差（定義は根拠参照） | 0.114285714 error | aggregate; n=35 | [E6](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・synthetic。不一致または非EOSのページ対/結果がある対。欠測は別計上。logitのnRMSEではない。本文差1件、本文一致で他の予測差3件。 |
| 参照との誤差（定義は根拠参照） | 0.0416666667 error | aggregate; n=24 | [E6](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・omnidocbench。不一致または非EOSのページ対/結果がある対。欠測は別計上。logitのnRMSEではない。本文差1件、本文一致で他の予測差0件。 |
| 参照との誤差（定義は根拠参照） | 0.0571428571 error | aggregate; n=35 | [E9](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・synthetic。不一致または非EOSのページ対/結果がある対。欠測は別計上。logitのnRMSEではない。本文差1件、本文一致で他の予測差1件。 |
| 参照との誤差（定義は根拠参照） | 0.166666667 error | aggregate; n=24 | [E9](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・omnidocbench。不一致または非EOSのページ対/結果がある対。欠測は別計上。logitのnRMSEではない。本文差2件、本文一致で他の予測差2件。 |
| 参照との誤差（定義は根拠参照） | 0.0571428571 error | aggregate; n=35 | [E12](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・synthetic。不一致または非EOSのページ対/結果がある対。欠測は別計上。logitのnRMSEではない。本文差1件、本文一致で他の予測差1件。 |
| 参照との誤差（定義は根拠参照） | 0.0416666667 error | aggregate; n=24 | [E12](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・omnidocbench。不一致または非EOSのページ対/結果がある対。欠測は別計上。logitのnRMSEではない。本文差0件、本文一致で他の予測差1件。 |
| 停止受付から完了までの時間 | 未計測 | — | 実モデルの対話的な取消時間は未計測。CPU故障試験とは別の範囲。 |
| 文字誤り率CER | 0.0167955029 ratio | mean; n=28 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| 文字誤り率CER | 0.0956737281 ratio | mean; n=16 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| 文字誤り率CER | 0.0167955029 ratio | mean; n=28 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| 文字誤り率CER | 0.0941209331 ratio | mean; n=16 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| 文字誤り率CER | 0.0167469759 ratio | mean; n=28 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| 文字誤り率CER | 0.0830689264 ratio | mean; n=16 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| 文字誤り率CER | 0.0167955029 ratio | mean; n=28 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| 文字誤り率CER | 0.0956737281 ratio | mean; n=16 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| 単語誤り率WER | 未計測 | — | 単語分割の定義は固定していない。CERを記録する。 |
| レイアウト・構造F1 | 未計測 | — | layout/構造F1は未計測。固定した読み順と表セル指標を別に記録する。 |
| 生成トークン数（layoutと全block） | 288 tokens | median; n=20 | [E1](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・F4・cache=prepared。layout・全block・EOSを含む。 |
| 生成トークン数（layoutと全block） | 288 tokens | median; n=20 | [E4](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・F4・cache=prepared。layout・全block・EOSを含む。 |
| 生成トークン数（layoutと全block） | 287 tokens | median; n=20 | [E7](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・F4・cache=prepared。layout・全block・EOSを含む。 |
| 生成トークン数（layoutと全block） | 288 tokens | median; n=20 | [E10](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・F4・cache=prepared。layout・全block・EOSを含む。 |
| OCR数値トークン再現率 | 0.992630385 ratio | mean; n=21 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン再現率 | 1 ratio | mean; n=6 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン再現率 | 0.992630385 ratio | mean; n=21 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン再現率 | 1 ratio | mean; n=6 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン再現率 | 0.996031746 ratio | mean; n=21 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン再現率 | 1 ratio | mean; n=6 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン再現率 | 0.992630385 ratio | mean; n=21 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン再現率 | 1 ratio | mean; n=6 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン適合率 | 0.992630385 ratio | mean; n=21 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン適合率 | 1 ratio | mean; n=6 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン適合率 | 0.992630385 ratio | mean; n=21 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン適合率 | 1 ratio | mean; n=6 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン適合率 | 0.996031746 ratio | mean; n=21 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン適合率 | 1 ratio | mean; n=6 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン適合率 | 0.992630385 ratio | mean; n=21 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR数値トークン適合率 | 1 ratio | mean; n=6 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR表セル順序付き一致率 | 0.986904762 ratio | mean; n=7 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR表セル順序付き一致率 | 0.857142857 ratio | mean; n=7 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR表セル順序付き一致率 | 0.986904762 ratio | mean; n=7 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR表セル順序付き一致率 | 0.857142857 ratio | mean; n=7 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR表セル順序付き一致率 | 0.979761905 ratio | mean; n=7 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR表セル順序付き一致率 | 0.851851852 ratio | mean; n=7 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR表セル順序付き一致率 | 0.986904762 ratio | mean; n=7 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR表セル順序付き一致率 | 0.857142857 ratio | mean; n=7 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・omnidocbench。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR正規化読み順誤り | 0 ratio | mean; n=7 | [E2](../results/teleocr-eval/2026-10-28/README.md) · coreai-fp16・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR正規化読み順誤り | 0 ratio | mean; n=7 | [E5](../results/teleocr-eval/2026-10-28/README.md) · coreai-int8・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR正規化読み順誤り | 0 ratio | mean; n=7 | [E8](../results/teleocr-eval/2026-10-28/README.md) · coreai-int4・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |
| OCR正規化読み順誤り | 0 ratio | mean; n=7 | [E11](../results/teleocr-eval/2026-10-28/README.md) · coreai-vision-int8・synthetic。当該指標がある入力のmacro平均。指標ごとのnと欠測/実行失敗は正本へ保持。#313の固定正規化/採点を使用。 |

### 検証した範囲

| 確認項目 | 状態 | 対象範囲・残る確認 | 根拠 |
|---|---|---|---|
| 変換・参照との一致 | 不合格（記載基準） | 4条件のページ出力厳密一致基準。条件別の一致/欠測数と未解釈の出力差を参照。数値部品検証と用途品質は別。 | [E3](../results/teleocr-eval/2026-10-28/README.md) · [E6](../results/teleocr-eval/2026-10-28/README.md) · [E9](../results/teleocr-eval/2026-10-28/README.md) · [E12](../results/teleocr-eval/2026-10-28/README.md) |
| 用途の品質 | 不合格（記載基準） | 固定自動採点のみ。条件付き/不足を保持。実写とネイティブアプリ受入は未確認。OmniDocBenchは選択した小subsetであり、未使用入力への一般化を保証しない。 | [E2](../results/teleocr-eval/2026-10-28/README.md) · [E5](../results/teleocr-eval/2026-10-28/README.md) · [E8](../results/teleocr-eval/2026-10-28/README.md) · [E11](../results/teleocr-eval/2026-10-28/README.md) |
| 連続利用・資源解放 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |
| 停止・再試行・回復 | 未計測 | この範囲の共通プロトコルによる確認は未実施です。 | — |

### 根拠と測定条件

| 根拠 | 日付・記録区分・経路 | 機種・OS・ツール | 入力と記録済みの条件 |
|---|---|---|---|
| [E1](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 合成F4ページ。cold1回、warmup1回、各条件を交互に新processで20回。画像復号・layout/crop・認識・表/Markdown組立を含み、file I/OとJSON serializationは除外。 · variant=coreai-fp16; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=1f3ab394969bb5c4a87e7653cc755080e92bf856909cb4ff637ad3c5182e1862; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=1; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E2](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 合成35ページと固定OmniDocBench領域24件を別採点、条件ごとに1回。元BF16・FP32対照・対応する圧縮FP32参照を別軸で比較。未使用実写の品質やネイティブアプリ受入を保証するものではない。 · variant=coreai-fp16; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=1f3ab394969bb5c4a87e7653cc755080e92bf856909cb4ff637ad3c5182e1862; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=0; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E3](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 同じ画像/ページ経路のFP32 PyTorch参照対Core AI。layout/box/prompt/text/OTSLとEOSの厳密一致基準。出力差を自動的に数値許容合格や変換器の欠陥の証明とはしない。 · variant=coreai-fp16; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true / PyTorch MPS; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=1f3ab394969bb5c4a87e7653cc755080e92bf856909cb4ff637ad3c5182e1862; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=0; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E4](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 合成F4ページ。cold1回、warmup1回、各条件を交互に新processで20回。画像復号・layout/crop・認識・表/Markdown組立を含み、file I/OとJSON serializationは除外。 · variant=coreai-int8; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=b9d084ac34fab6c99e760b6debff657057b4358f23f02ee1041638b68d904429; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=1; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E5](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 合成35ページと固定OmniDocBench領域24件を別採点、条件ごとに1回。元BF16・FP32対照・対応する圧縮FP32参照を別軸で比較。未使用実写の品質やネイティブアプリ受入を保証するものではない。 · variant=coreai-int8; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=b9d084ac34fab6c99e760b6debff657057b4358f23f02ee1041638b68d904429; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=0; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E6](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 同じ画像/ページ経路のFP32 PyTorch参照対Core AI。layout/box/prompt/text/OTSLとEOSの厳密一致基準。出力差を自動的に数値許容合格や変換器の欠陥の証明とはしない。 · variant=coreai-int8; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true / PyTorch MPS; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=b9d084ac34fab6c99e760b6debff657057b4358f23f02ee1041638b68d904429; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=0; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E7](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 合成F4ページ。cold1回、warmup1回、各条件を交互に新processで20回。画像復号・layout/crop・認識・表/Markdown組立を含み、file I/OとJSON serializationは除外。 · variant=coreai-int4; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=e85b0be668cccdcd0a0114b4f9713baec421b16d4a487dfc582655954ac07b66; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=1; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E8](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 合成35ページと固定OmniDocBench領域24件を別採点、条件ごとに1回。元BF16・FP32対照・対応する圧縮FP32参照を別軸で比較。未使用実写の品質やネイティブアプリ受入を保証するものではない。 · variant=coreai-int4; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=e85b0be668cccdcd0a0114b4f9713baec421b16d4a487dfc582655954ac07b66; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=0; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E9](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 同じ画像/ページ経路のFP32 PyTorch参照対Core AI。layout/box/prompt/text/OTSLとEOSの厳密一致基準。出力差を自動的に数値許容合格や変換器の欠陥の証明とはしない。 · variant=coreai-int4; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true / PyTorch MPS; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=e85b0be668cccdcd0a0114b4f9713baec421b16d4a487dfc582655954ac07b66; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=0; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E10](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 合成F4ページ。cold1回、warmup1回、各条件を交互に新processで20回。画像復号・layout/crop・認識・表/Markdown組立を含み、file I/OとJSON serializationは除外。 · variant=coreai-vision-int8; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=b7de99594ace296ba44b7623832da3a75c3b1e06de5daa415236eaa4d49b5ea7; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=1; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E11](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 合成35ページと固定OmniDocBench領域24件を別採点、条件ごとに1回。元BF16・FP32対照・対応する圧縮FP32参照を別軸で比較。未使用実写の品質やネイティブアプリ受入を保証するものではない。 · variant=coreai-vision-int8; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=b7de99594ace296ba44b7623832da3a75c3b1e06de5daa415236eaa4d49b5ea7; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=0; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |
| [E12](../results/teleocr-eval/2026-10-28/README.md) | 2026-10-28 / standard / diagnostic | Apple M4 Max / 137438953472 B RAM / macOS 27.0 (26A428) / Xcode 27.0; Build version 27A266a; Apple Swift version 6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1) | 同じ画像/ページ経路のFP32 PyTorch参照対Core AI。layout/box/prompt/text/OTSLとEOSの厳密一致基準。出力差を自動的に数値許容合格や変換器の欠陥の証明とはしない。 · variant=coreai-vision-int8; backend=Core AI SpecializationOptions.default, expectFrequentReshapes=true / PyTorch MPS; route=coreai-verify ocr; two-stage page pipeline; source_revision=70a19ec6a3b036311af19ae5e762e650f41a5980; artifact_revision=b7de99594ace296ba44b7623832da3a75c3b1e06de5daa415236eaa4d49b5ea7; code_revision=ea722fca17aaf0a469d9d0cbf8271b7c3d348e8d; batch_size=1; concurrency=1; warmups=0; power=AC Power (performance runs sampled before/after; quality runs share the invocation); thermal=Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; no GPU utilization or temperature measurement |

### 導入・制限・進捗

- [#178](https://github.com/oka-yuji/coreai-samples/issues/178)
<!-- model-report:end -->
