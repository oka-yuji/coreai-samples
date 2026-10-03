# Core AI Samples

This repository will hold a Swift runtime, a verification harness, and a sample app for running open LLMs on Apple silicon with **Core AI** and **Foundation Models**.

> Work in progress. Nothing here is released yet.

## Requirements

- macOS 27 or iOS 27
- Xcode 27 with the Metal Toolchain component
- [XcodeGen](https://github.com/yonaskolb/XcodeGen), only to regenerate the demo app's project

## Demo app (macOS)

`Examples/DemoApp` is a SwiftUI app that runs a local model bundle through Foundation Models with the runtime in this repository (`CoreAIBackend`). It does not use the official `coreai-models` package.

- **Models**: open a bundle folder on this Mac (the app remembers it), load it, and clear the copy the app prepared for the GPU. The download catalog for Hugging Face is in place but stays empty until bundles are published.
- **Chat**: a `LanguageModelSession` on `LocalLanguageModel`. Replies stream, thinking streams into a collapsed section, and the status line shows the time to the first token, the speed, the input, reused and output tokens, and why the reply stopped.

### Build and run

```bash
cd Examples/DemoApp
xcodegen generate   # only needed after editing project.yml; the generated project is committed
xcodebuild -scheme DemoApp -configuration Release -derivedDataPath .build build
open .build/Build/Products/Release/DemoApp.app
```

The app is signed ad hoc and runs without the App Sandbox. To sign with your team, copy `Local.xcconfig.template` to `Local.xcconfig` and set `DEVELOPMENT_TEAM`. To try a small context window, launch it with `open .build/Build/Products/Release/DemoApp.app --args --context-limit 1024`.

The first load of a model prepares it for the GPU (about 21 s for Qwen3.5-9B on an M4 Max); later loads reuse the prepared copy in `~/Library/Caches/coreai-cache/<OS build>/com.coreai-samples.DemoApp/`.

### Headless self-test

```bash
APP=.build/Build/Products/Release/DemoApp.app/Contents/MacOS/DemoApp
$APP --selftest --model /path/to/bundle --prompts-file prompts.txt [--max-tokens 2048] [--context-limit 4096]
```

- `prompts.txt` holds the turns of one conversation, separated by lines that read `===PROMPT===`.
- Each turn prints one JSON line: `turn`, `answer`, `inputTokens`, `cachedTokens` (reused), `outputTokens`, `ttftSeconds`, `tokensPerSecond` (output tokens divided by the decode time) and `stopReason`. The last line is `SELFTEST PASS` or `SELFTEST FAIL`, and the exit status is 0 or 1.
- A pass means every turn finished without an error, generated tokens, stopped on a stop token or the token limit, and every turn after the first reused tokens. No window opens.

### Screen check

```bash
$APP --gui-e2e /path/to/output --model /path/to/bundle [--clear-prepared] [--context-limit 1024]
```

The app opens the bundle in Models, loads it, chats one turn, starts a long reply, stops it and starts a new chat, saving a screenshot after each step with `results.json`, `driver.log` and the metrics rows of the run. `--clear-prepared` removes the prepared copy first, so the load shows the preparing state. With `--context-limit`, it then keeps chatting until the context window is full.

### Metrics

Every generation appends one JSON line to `~/Library/Application Support/com.coreai-samples.DemoApp/metrics.jsonl`: the model, the turn, the input, reused and output tokens, the time to the first token, the speed, the stop reason, the resident memory and the app build.

## Related

- [coreml-llm-samples](https://github.com/oka-yuji/coreml-llm-samples) — the Core ML line, for macOS 26 / iOS 26.

## License

Code: MIT — see [LICENSE](LICENSE). Model weights are distributed separately under their own licenses.
