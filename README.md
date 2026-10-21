# Core AI Samples

[日本語](README.ja.md)

Swift samples for running open models with Apple's Core AI and Foundation Models. Explore local **chat and research**, **document decisions and extraction**, **image search**, and **speech**, with conversion recipes and recorded verification results.

Start with a model below. Each guide covers setup, supported devices, measurements and limitations; the status describes the converted bundle's public availability, separately from app integration.

## Models and roadmap

**Released** means a converted bundle is publicly downloadable; **In progress** includes conversion, verification and release preparation; **Planned** means agreed work has not started. Entries marked *investigation* are feasibility studies, not a commitment to release or device support.

The GLiNER2.5-multi and EmbeddingGemma 2 repositories are public and anonymously downloadable. The Qwen3.8 27B and Clef-Flash 9B repositories still contain **private bundles** and require access. A completed conversion or merged app implementation alone does not count as a public release. Model-specific guides and issues record the verified scope.

<!-- model-catalog:start -->
Checked: **2026-10-19**. ✅ = recorded native-app execution on that OS for the variant/route in the guide; △ = implementation, older diagnostic evidence or acceptance pending; — = no native demo. A mark is separate from bundle publication, conversion success and full production qualification.

| Model / bundle | Use | macOS | iOS | Status | Download | Guide |
|---|---|---|---|---|---|---|
| Qwen3.8 27B · 4-bit, text | Chat, Work, `coreai-chat` | [✅](docs/qwen3.8-27b.md#model-report) | [—](docs/qwen3.8-27b.md#model-report) | In progress · private bundle | [Access required](https://huggingface.co/okayuji/Qwen3.8-27B-coreai-4bit) | [Guide / results](docs/qwen3.8-27b.md) |
| Clef-Flash 9B · uint4 + fp16 head, text | Recipes, Decide | [✅](docs/clef-flash.md#model-report) | [—](docs/clef-flash.md#model-report) | In progress · private bundle | [Access required](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide) | [Guide / results](docs/clef-flash.md) |
| GLiNER2.5-multi · FP32 | Product extraction, inspection memos | [△](docs/model-reports/gliner25.md#model-report) | [△](docs/model-reports/gliner25.md#model-report) | Released · iOS device acceptance pending | [Hugging Face](https://huggingface.co/okayuji/gliner2.5-multi-coreai-fp32) | [Guide / results](docs/model-reports/gliner25.md) |
| EmbeddingGemma 2 · FP16 / INT4-INT8 PTQ | Image search, video moment search, camera object search | [✅](docs/embeddinggemma2.md#model-report) | [△](docs/embeddinggemma2.md#model-report) | Released · iOS device acceptance pending | [Hugging Face](https://huggingface.co/okayuji/embeddinggemma-2-coreai) | [Guide / results](docs/embeddinggemma2.md) |
| Irodori · FP32 | Speech generation and playback | [△](docs/model-reports/irodori.md#model-report) | [△](docs/model-reports/irodori.md#model-report) | In progress · app/device acceptance | — | [Guide / results](docs/model-reports/irodori.md) |
| Qwen Image 2.1 Turbo | Image generation and editing | [—](docs/model-reports/qwen-image21.md#model-report) | [—](docs/model-reports/qwen-image21.md#model-report) | In progress · Mac conversion verified; mobile implementation in progress | — | [Guide / results](docs/model-reports/qwen-image21.md) |
| Magpie TTS Multilingual v2607 | Speech generation | [—](docs/model-reports/magpie-tts.md#model-report) | [—](docs/model-reports/magpie-tts.md#model-report) | In progress · conversion verified; listening / publication pending | — | [Guide / results](docs/model-reports/magpie-tts.md) |
| Qwen3-TTS CustomVoice · 0.6B | Speech generation | [—](docs/model-reports/qwen3-tts-06b.md#model-report) | [—](docs/model-reports/qwen3-tts-06b.md#model-report) | In progress · conversion verified; listening / publication pending | — | [Guide / results](docs/model-reports/qwen3-tts-06b.md) |
| Qwen3-TTS CustomVoice · 1.7B | Speech generation | [—](docs/model-reports/qwen3-tts-17b.md#model-report) | [—](docs/model-reports/qwen3-tts-17b.md#model-report) | In progress · conversion verified; listening / publication pending | — | [Guide / results](docs/model-reports/qwen3-tts-17b.md) |
| TeleOCR | Document OCR | [—](docs/model-reports/teleocr.md#model-report) | [—](docs/model-reports/teleocr.md#model-report) | Planned · investigation | — | [Guide / results](docs/model-reports/teleocr.md) |
| MiniCPM-V 4.7 1B | Image/video understanding | [—](docs/model-reports/minicpm-v47.md#model-report) | [—](docs/model-reports/minicpm-v47.md#model-report) | Planned · investigation | — | [Guide / results](docs/model-reports/minicpm-v47.md) |
| Depth Anything 3 Small | Depth and camera pose | [—](docs/model-reports/depth-anything3.md#model-report) | [—](docs/model-reports/depth-anything3.md#model-report) | Planned · investigation | — | [Guide / results](docs/model-reports/depth-anything3.md) |
| LTX 2.5 | Video generation | [—](docs/model-reports/ltx25.md#model-report) | [—](docs/model-reports/ltx25.md#model-report) | Planned · investigation | — | [Guide / results](docs/model-reports/ltx25.md) |
<!-- model-catalog:end -->

The iOS app opens **Models**, with six demos connected to their providers. Models downloads the pinned GLiNER2.5-multi FP32 bundle and the EmbeddingGemma 2 FP16 (default) or INT4/INT8 PTQ bundle anonymously from Hugging Face, and a demo runs only after every file's size and SHA-256 has been checked ([#210](https://github.com/oka-yuji/coreai-samples/issues/210)). See the [integration status](docs/validation/ios-models-integration-20261018.md); running the six demos on an iPhone remains a user check. Local diagnostic runs and a bundle's public release are separate milestones.

## Camera search demo

Choose what to find by typing a target, then change the text to switch targets without additional training. [Watch the EmbeddingGemma 2 camera search demo on X](https://x.com/oka_yuuji/status/2111307591586578612). See the [iPhone guide](docs/object-finder.md) for setup and limitations.

## Get started

Development requires an **Apple silicon Mac, Xcode 27 and the Metal Toolchain**. The Mac demos require macOS 27; iPhone demos require a supported physical device with iOS 27. Core AI execution is unavailable in the iOS simulator. Model memory requirements and verified device coverage are in each guide.

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
```

Install Rust/Cargo for the speech input adapter and Python 3.9+ for repository checks. Open `CoreAISamples.xcodeproj`, choose **CoreAISamples** and **My Mac**, then run. The scheme builds Release; **CoreAISamples-Debug** is available for debugging. Follow a model's guide to obtain and select its bundle. Weights are separate from this repository, and private bundles require an authorized Hugging Face account.

For an iPhone build, signing, or the separate speech app, follow the [app setup guide](docs/demo-app.md#iphone-app). A successful build does not establish that every model works on that device.

## Developer guides

- [Common model evaluation and reporting](docs/benchmarking.md): metrics, measurement plans, evidence records and generated tables.

- [Swift API and language-generation runtime](docs/runtime.md): converting your own bundle, CLI, supported requests, cache and context behavior, and repository layout.
- [Verification and measurements](docs/verification.md): build/test commands, runtime comparisons and the conditions behind results.
- [Demo app](docs/demo-app.md): building, headless checks, screen checks and logs.
- [Conversion tools](Tools/Conversion/README.md): reproducible conversion recipes and maintenance workflow.
- [Model-first specification](docs/specs/model-first-demo.html), [iOS design](docs/designs/model-first-ios.html), [macOS design](docs/designs/model-first-macos.html), and [feature-screen designs](docs/designs/function-screens.html): proposals and implementation tracking. These HTML documents are in Japanese; open the local files in a browser for the formatted view.

## Related

- [coreml-llm-samples](https://github.com/oka-yuji/coreml-llm-samples) — the Core ML line, for macOS 26 / iOS 26.

## License

Code: MIT — see [LICENSE](LICENSE). Model weights have their own licenses; use each guide and bundle's license/notice files. Qwen3.8 27B and Clef-Flash 9B retain their source models' Apache License 2.0, with `LICENSE` and `NOTICE.md` in the respective Hugging Face repositories. GLiNER2.5-multi and EmbeddingGemma 2 retain their source models' Apache License 2.0; each bundle includes `LICENSE`, with `THIRD_PARTY_NOTICES.md` (GLiNER2.5-multi, including the MIT notice of its mDeBERTa encoder and tokenizer) or `MODEL_LICENSE.txt` (EmbeddingGemma 2).
