# Core AI Samples

[日本語](README.ja.md)

Swift samples for running open models with Apple's Core AI and Foundation Models. Explore local **chat and research**, **document decisions and extraction**, **image search**, and **speech**, with conversion recipes and recorded verification results.

Start with a model below. Each guide covers setup, supported devices, measurements and limitations; the status describes the converted bundle's public availability, separately from app integration.

## Models and roadmap

Last checked: **2026-10-19**. **Released** means a converted bundle is publicly downloadable; **In progress** includes conversion, verification and release preparation; **Planned** means agreed work has not started. Entries marked *investigation* are feasibility studies, not a commitment to release or device support.

The four linked Hugging Face repositories currently contain **private bundles** and require access. None is anonymously downloadable at this checkpoint. A completed conversion or merged app implementation alone does not count as a public release. Model-specific guides and issues record the verified scope.

| Model / bundle | Use | Status | Download | Guide / progress |
|---|---|---|---|---|
| Qwen3.8 27B · 4-bit, text | Chat, Work, `coreai-chat` · macOS | In progress · private bundle | [Access required](https://huggingface.co/okayuji/Qwen3.8-27B-coreai-4bit) | [Chat and Work](docs/qwen3.8-27b.md) |
| Clef-Flash 9B · uint4 + fp16 head, text | Recipes, Decide · macOS | In progress · private bundle | [Access required](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide) | [Recipes and Decide](docs/clef-flash.md) |
| GLiNER2.5-multi · FP32 | Product extraction, inspection memos | In progress · public release / iOS adoption | [Access required](https://huggingface.co/okayuji/gliner2.5-multi-coreai-fp32) | [Product demo](docs/specs/product-comparison-demo.md) · [#210](https://github.com/oka-yuji/coreai-samples/issues/210) |
| EmbeddingGemma 2 · FP16 / INT4-INT8 PTQ | Image search, camera object search | In progress · public release / iOS adoption | [Access required](https://huggingface.co/okayuji/embeddinggemma-2-coreai) | [Search guide](docs/embeddinggemma2.md) · [#210](https://github.com/oka-yuji/coreai-samples/issues/210) |
| Irodori · FP32 | Speech generation and playback | In progress · app/device acceptance | — | [Mac](docs/irodori/demo.md) · [iPhone](docs/irodori/iphone.md) · [#93](https://github.com/oka-yuji/coreai-samples/issues/93) |
| Qwen Image 2.1 Turbo · macOS FP32 | Image generation and editing | In progress · Mac conversion verified; mobile planned | — | [Conversion](Tools/Conversion/recipes/qwen_image21/README.md) · [Mobile plan #222](https://github.com/oka-yuji/coreai-samples/issues/222) |
| Magpie TTS Multilingual v2607 | Speech generation | In progress · conversion and verification | — | [#165](https://github.com/oka-yuji/coreai-samples/issues/165) · [PR #166](https://github.com/oka-yuji/coreai-samples/pull/166) |
| Qwen3-TTS CustomVoice · 0.6B | Speech generation | In progress · conversion and verification | — | [#167](https://github.com/oka-yuji/coreai-samples/issues/167) · [PR #168](https://github.com/oka-yuji/coreai-samples/pull/168) |
| Qwen3-TTS CustomVoice · 1.7B | Speech generation | In progress · conversion and verification | — | [#169](https://github.com/oka-yuji/coreai-samples/issues/169) · [PR #170](https://github.com/oka-yuji/coreai-samples/pull/170) |
| TeleOCR | Document OCR | Planned · investigation | — | [#178](https://github.com/oka-yuji/coreai-samples/issues/178) |
| MiniCPM-V 4.7 1B | Image/video understanding | Planned · investigation | — | [#179](https://github.com/oka-yuji/coreai-samples/issues/179) |
| Depth Anything 3 Small | Depth and camera pose | Planned · investigation | — | [#180](https://github.com/oka-yuji/coreai-samples/issues/180) |
| LTX 2.5 | Video generation | Planned · investigation | — | [#181](https://github.com/oka-yuji/coreai-samples/issues/181) |

The iOS app opens **Models**, with five demos connected to their providers. Their model distribution definitions are being adopted in [#210](https://github.com/oka-yuji/coreai-samples/issues/210); until that lands, the normal Models route shows descriptions without download or execution. See the [integration status](docs/validation/ios-models-integration-20261018.md). Local diagnostic runs and a bundle's public release are separate milestones.

## Get started

Development requires an **Apple silicon Mac, Xcode 27 and the Metal Toolchain**. The Mac demos require macOS 27; iPhone demos require a supported physical device with iOS 27. Core AI execution is unavailable in the iOS simulator. Model memory requirements and verified device coverage are in each guide.

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
```

Install Rust/Cargo for the speech input adapter. Open `CoreAISamples.xcodeproj`, choose **CoreAISamples** and **My Mac**, then run. The scheme builds Release; **CoreAISamples-Debug** is available for debugging. Follow a model's guide to obtain and select its bundle. Weights are separate from this repository, and private bundles require an authorized Hugging Face account.

For an iPhone build, signing, or the separate speech app, follow the [app setup guide](docs/demo-app.md#iphone-app). A successful build does not establish that every model works on that device.

## Developer guides

- [Swift API and language-generation runtime](docs/runtime.md): converting your own bundle, CLI, supported requests, cache and context behavior, and repository layout.
- [Verification and measurements](docs/verification.md): build/test commands, runtime comparisons and the conditions behind results.
- [Demo app](docs/demo-app.md): building, headless checks, screen checks and logs.
- [Conversion tools](Tools/Conversion/README.md): reproducible conversion recipes and maintenance workflow.
- [Model-first specification](docs/specs/model-first-demo.html), [iOS design](docs/designs/model-first-ios.html), [macOS design](docs/designs/model-first-macos.html), and [feature-screen designs](docs/designs/function-screens.html): proposals and implementation tracking. These HTML documents are in Japanese; open the local files in a browser for the formatted view.

## Related

- [coreml-llm-samples](https://github.com/oka-yuji/coreml-llm-samples) — the Core ML line, for macOS 26 / iOS 26.

## License

Code: MIT — see [LICENSE](LICENSE). Model weights have their own licenses; use each guide and bundle's license/notice files. Qwen3.8 27B and Clef-Flash 9B retain their source models' Apache License 2.0, with `LICENSE` and `NOTICE.md` in the respective Hugging Face repositories.
