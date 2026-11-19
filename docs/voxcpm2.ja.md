# VoxCPM2 の音声合成のデモ

[ホーム](../README.ja.md) · [English](voxcpm2.md)

Mac のアプリの **VoxCPM2** の画面は、[openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)(Apache-2.0)の FP16 の Core AI の bundle で、文章から音声を生成します。文章には声質の指定を書き添えられます。サイドバーの **VoxCPM2** の見出しからモデルをダウンロードし、その下の行を選んで、生成・再生・停止と、48 kHz・モノラル・Float32 の WAV の保存を行います。iPhone の版はありません。

アプリは、公開した bundle [okayuji/voxcpm2-coreai](https://huggingface.co/okayuji/voxcpm2-coreai) の `fp16` のフォルダを、固定した revision `afe11d61de9dd7c3609140aa040b777ef77b2555` からダウンロードします。bundle は[変換の手順](../Tools/Conversion/recipes/voxcpm2/README.md)で作りました。変換の照合・時間・メモリの実測は[変換の検証記録](validation/voxcpm2-conversion-20261117.md)にあります。2026-11-18 に本人が、Mac の通常の画面でダウンロード・生成・再生・保存ができ、声に問題が無いことを報告しました([受け入れの記録](validation/voxcpm2-mac-acceptance-20261118.md))。音質の点数の記録はありません。

## モデルをダウンロードする

1. **VoxCPM2** の見出しの「モデルをダウンロード」を押します。アプリは、固定した 22 ファイル(合計 4,677,413,664 バイト、約 4.7 GB)を、サインインせずにダウンロードします。全ファイルの大きさと SHA-256、manifest の構成を確かめてから、モデルを準備済みと表示します。後でモデルを開くときに、アプリのキャッシュに検査済みの複製をもう 1 つ作るので、両方の空き容量を用意します。
2. ダウンロードできるのはこの FP16 の構成だけです。FP32 の bundle と、projection だけ FP32 の FP16 の bundle は出ません。見出しにフォルダを取り込む操作はありません。
3. 前の版のアプリで FP16 の bundle を取り込んでいた場合は、ダウンロードし直さずに、見出しに準備済みと表示されます。ダウンロードは、その取込と同じ識別子・フォルダ・モデルの領域の記録を使います。識別子は公開先の revision に追従しません。後の revision でも、構成と manifest が同じなら、同じフォルダを使います。

後の版のアプリで固定したファイルの一覧が変わると、前のコピーが準備済みと表示されたまま、開くと誤りになることがあります。記録済みのコピーの通常の確認は、ファイルの有無と大きさを見て、中身は見ません。そのため、大きさが同じまま中身が変わったコピーも準備済みと表示され、画面が開くときの全ファイルの検査で誤りになります。見出しから削除して、ダウンロードし直します。

## 画面を使う

1. **VoxCPM2** の行を選びます。画面はダウンロードしたモデルを自分で開き、全ファイルをもう一度確かめます(モデルは動かしません)。モデルが無ければ、サイドバーの見出しからダウンロードするよう案内します。
2. 例文を選ぶか、文章を入力します。声質は、文の欄の下の **Voice description**(日本語の表示では「声質」)の欄に短く書きます。例: `A young woman, gentle and calm voice`。画面は、説明から前後の空白と半角・全角の括弧を除き、残りを半角の括弧で囲んで文の前に付け、モデルへ渡します。文は変えません。説明が残らなければ、モデルへは文だけを渡します。声質にひらがなかカタカナがあると、画面はこの Mac の中で声質を英語に翻訳してから、訳を同じように文の前に付けます。英語・中国語・漢字だけの声質は、そのまま渡します。読み上げる文は翻訳しません。モデルへ渡した文は、生成の後に **Text used for generation**(日本語の表示では「生成に使用したテキスト」)で確かめられます。翻訳用の言語(日本語と英語)がこの Mac に入っていないときは、生成せずに、システム設定で言語をダウンロードするか声質を英語で書くよう案内します。訳の文は OS の版で変わることがあり、同じ入力と同じ seed でも、訳が変われば音声も変わることがあります。欄が空なら、文の先頭に半角の括弧で書いた説明(例: `(A young woman, gentle and calm voice)確認が終わりました。`)も使えます。欄にも文の先頭にも説明が無いときは、声はモデルが決めます。例文を選ぶと文と声質の両方が置き換わり、**Clear** は文だけを消します。例文は、説明のない文、説明つきの日本語の文、説明つきの英語の文です。
3. 文と声質を合わせて、text token は 144 個までです。付け足した括弧を含めて組み立てた文を、空白を整えて複数の漢字の token を分けた後に数えます。長い入力は切り詰めずに拒否します。モデルが読めない文字を含む入力も拒否し、その文字を消すか置き換えるよう案内します。
4. **Generate speech** を押します。モデルを開いた直後、画面に戻った後、失敗の後の最初の生成には、モデルの資産を読み込む時間も入ります。音声が終わる前にモデルの長さの上限に達した生成は誤りとして表示し、音声を返しません。seed は符号なし 64 ビットの整数で、最初の値は `1708` です。同じ入力と同じ seed なら同じ音声になります(翻訳した声質では、訳が同じ間)。seed を変えると、別の読み上げを試せます。
5. **Play**・**Stop**・**Save WAV…** は Irodori の画面と同じです。保存の既定のファイル名は `voxcpm2-speech.wav` です。

画面を離れると、進行中の処理を止めてモデルを解放します。画面を表示している間は、見出しからモデルを削除できません。VoxCPM2 と Irodori の画面は、文・seed・音声・モデルのフォルダを別々に持ちます。

## 含まないもの

参照音声(声のクローン)、逐次の再生、音量をそろえる処理、ほかの構成の bundle、画面からのフォルダの取込、iPhone。

## 確認

- `swift test --package-path CoreAISamplesKit` は、カタログとダウンロードの定義(`MacDemoCatalogTests`、`VoxCPMDownloadTests`)、生成のボタンの条件(`SpeechGenerationGateTests`)、画面の Store(`SpeechStoreTests`)、声質と文の組み立て(`SpeechVoiceTests`)、かなを含む声質の翻訳の経路(`VoiceTranslationFeatureTests`、`VoiceTranslationRunnerTests`)、文の誤り(`VoxCPMTextErrorTests`)を確かめます。アプリの `--check-localization` は、サイドバーとカタログの対応と、日本語の文字列を確かめます。
- `VOXCPM2_TOKENIZER=/path/to/fp16 swift test --package-path CoreAISamplesKit --force-resolved-versions --filter SpeechRuntimeTests.VoxCPMTokenizerTests` は、例文の text token の数も数えます。読むのは `tokenizer.json` だけです。
- `COREAI_VOICE_TRANSLATION_TEST=1 swift test --package-path CoreAISamplesKit --force-resolved-versions --filter VoiceTranslationRunnerTests` は、この Mac の翻訳を実際に 1 回呼び、訳が空でなく、かなを含まないことを確かめます。翻訳用の言語(日本語と英語)が入っている Mac で動かします。モデルは読みません。既定の実行では skip します。
- `AGENTS.md` にあるダウンロードの試験は、共有の heavy-job ロックの下で動かします。固定したファイルを、認証なしで空の一時のモデルの領域へダウンロードし、全ファイルを確かめ、利用の lease を取り、本番のサービスと Store で例文を 1 つ生成し、停止し、lease を返し、コピーを削除します。
- `AGENTS.md` にある取込の通しの試験は、画面には出なくなった取込の経路で、ローカルの FP16 の bundle を同じモデルの領域へ取り込み、全部の例文を生成します。
- これらの確認は、聴いたときの品質と通常の画面での操作を保証しません。2026-11-18 に本人が、アプリでダウンロード・生成・再生・保存ができ、声に問題が無いことを報告しました。報告の範囲は[受け入れの記録](validation/voxcpm2-mac-acceptance-20261118.md)にあります。
