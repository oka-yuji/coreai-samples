# Mac model distribution (2026-11-04)

2026-11-04 JST, Issue #464 (parent #458). The Hugging Face repositories [okayuji/Qwen3.8-27B-coreai-4bit](https://huggingface.co/okayuji/Qwen3.8-27B-coreai-4bit) and [okayuji/clef-flash-9b-coreai-decide](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide) changed from private to public on 2026-11-04, after the [information audit](#information-audit) passed. This record shows that both repositories, and every file of their pinned revisions, are reachable without signing in.

The check ran in Python with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and no token in the environment. `HfApi(token=False)` read each repository's refs and model information with file metadata. Every file at the pinned revision then received an HTTP `HEAD` request on its resolve URL, without an `Authorization` header and following redirects (307 for regular files and 302 for LFS files, then 200). The tables list the final status, the `x-repo-commit` header and the Hub size, which equaled the response's `Content-Length`. No request carried an `Authorization` header (0 of 141 HTTP exchanges, redirects included). LFS files were not downloaded; their SHA-256 values are the Hub's LFS metadata. The other files were downloaded and hashed, and each matched the Hub's size and git blob ID. The [machine-readable record](mac-models-distribution-20261104/distribution.json) lists each file's size, SHA-256 and source (`lfs` or `downloaded`) with the HEAD results. No model was loaded, and the app's downloader was not used.

## Qwen3.8 27B

- Repository: [okayuji/Qwen3.8-27B-coreai-4bit](https://huggingface.co/okayuji/Qwen3.8-27B-coreai-4bit), public and not gated.
- Refs: branch `main` is `843c3846b683eb2c946377af2c4d1aed9099278d`, and tag `v1.0.0` is `8e24aebe036eae577d4a3bd019ba728a73358318`. The app catalog and the [guide](../qwen3.8-27b.md#quick-start) pin `v1.0.0`.
- Pinned revision `8e24aebe036eae577d4a3bd019ba728a73358318`: 12 files, 16,852,543,780 bytes, equal to the catalog size. Two files are LFS.
- `main` also has 12 files, totaling 16,852,544,376 bytes. Only `README.md` differs: 23,057 bytes at `v1.0.0` and 23,653 bytes on `main`. The other 11 files have the same sizes and object IDs.
- Checked at 2026-11-04 04:51 JST on macOS 27.2 (26B123), arm64, with Python 3.11.9, huggingface_hub 0.36.2 and requests 2.33.1. All 12 files answered HEAD with 200, the Hub size and the pinned revision.

| Path | Bytes | HEAD status | `x-repo-commit` |
|---|---:|---|---|
| `.gitattributes` | 1,675 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `LICENSE` | 11,544 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `NOTICE.md` | 2,031 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `README.md` | 23,057 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `SHA256SUMS` | 742 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `metadata.json` | 611 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `qwen3_8_27b_qwen3_5_cfg_gdnfp32_dynamic.aimodel/main.hash` | 32 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `qwen3_8_27b_qwen3_5_cfg_gdnfp32_dynamic.aimodel/main.mlirb` | 16,839,684,298 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `qwen3_8_27b_qwen3_5_cfg_gdnfp32_dynamic.aimodel/metadata.json` | 394 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `tokenizer/chat_template.jinja` | 8,952 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `tokenizer/tokenizer.json` | 12,809,320 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| `tokenizer/tokenizer_config.json` | 1,124 | 200 | `8e24aebe036eae577d4a3bd019ba728a73358318` |
| **Total (12 files)** | **16,852,543,780** | | |

## Clef-Flash 9B

- Repository: [okayuji/clef-flash-9b-coreai-decide](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide), public and not gated.
- Refs: branch `main` is `12e30be4e391310377c534524df8962359a08b0f`; there are no tags. The [guide](../clef-flash.md#quick-start) pins this revision.
- Pinned revision `12e30be4e391310377c534524df8962359a08b0f`: 25 files, 7,278,592,396 bytes. Four files are LFS.
- Checked at 2026-11-04 04:51 JST in the same environment. All 25 files answered HEAD with 200, the Hub size and the pinned revision.

| Path | Bytes | HEAD status | `x-repo-commit` |
|---|---:|---|---|
| `.gitattributes` | 1,792 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `LICENSE` | 11,544 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `NOTICE.md` | 2,402 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `README.md` | 11,648 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `SHA256SUMS` | 1,519 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `VERIFICATION.md` | 4,782 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `backbone/clef_flash_9b_hidden_gdnfp32_qwen3_5_cfg_dynamic.aimodel/main.hash` | 32 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `backbone/clef_flash_9b_hidden_gdnfp32_qwen3_5_cfg_dynamic.aimodel/main.mlirb` | 4,980,751,116 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `backbone/clef_flash_9b_hidden_gdnfp32_qwen3_5_cfg_dynamic.aimodel/metadata.json` | 105 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `backbone/lexical.safetensors` | 2,034,237,672 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `backbone/metadata.json` | 639 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `backbone/tokenizer/chat_template.jinja` | 7,756 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `backbone/tokenizer/tokenizer.json` | 19,989,325 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `backbone/tokenizer/tokenizer_config.json` | 1,124 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `decision.json` | 2,224 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `examples/maintenance-request.json` | 9,179 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `examples/maintenance-response.json` | 2,454 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `head/clef_flash_9b_head_q16o8_fp16.aimodel/main.hash` | 32 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `head/clef_flash_9b_head_q16o8_fp16.aimodel/main.mlirb` | 243,542,718 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `head/clef_flash_9b_head_q16o8_fp16.aimodel/metadata.json` | 105 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `verification/alternating-question-sizes.json` | 3,577 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `verification/app-scenario.json` | 424 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `verification/context-reuse.json` | 2,488 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `verification/gui-and-cancellation.json` | 4,380 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| `verification/ten-document-summary.json` | 3,359 | 200 | `12e30be4e391310377c534524df8962359a08b0f` |
| **Total (25 files)** | **7,278,592,396** | | |

## Information audit

Both repositories passed the model-release information audit (scanner `63501ba4`) before publication. The local check reads every byte of the staged payload. The hub check reads every object reachable from all refs, including pull-request refs, over the full history. Both look for personal paths in UTF-8 and UTF-16 and for common credential and private-key patterns. The gate confirms that the scanner, the exception set, the payload hashes, the remote refs and the visibility still match those checks, and that the local and remote `main` manifests are equal. The audit is not a complete secret classifier, and a pass applies only to the checked input and scope. The reports are kept outside the repository.

| Model | Check | Time (JST) | Scope | Result |
|---|---|---|---|---|
| Qwen3.8 27B | local | 2026-11-03 15:41 | 12 files, 16,852,544,376 bytes (the `main` payload) | passed |
| Qwen3.8 27B | hub, authenticated, while private | 2026-11-04 00:45 | 2 refs (`main`, `v1.0.0`), 5 commits, 16 unique objects | passed |
| Qwen3.8 27B | gate | 2026-11-04 01:21 | 2 refs; `main` manifest equal to the local payload | passed |
| Qwen3.8 27B | hub, anonymous, after publication | 2026-11-04 05:17 | 2 refs (`main`, `v1.0.0`), 5 commits, 16 unique objects; `main` manifest equal to the check before publication | passed |
| Clef-Flash 9B | local | 2026-11-04 00:20 | 25 files, 7,278,592,396 bytes | passed |
| Clef-Flash 9B | hub, authenticated, while private | 2026-11-04 00:35 | 1 ref (`main`), 18 commits, 30 unique objects | passed |
| Clef-Flash 9B | gate | 2026-11-04 00:51 | 1 ref; `main` manifest equal to the local payload | passed |
| Clef-Flash 9B | hub, anonymous, after publication | 2026-11-04 01:14 | 1 ref (`main`), 18 commits, 30 unique objects; `main` manifest equal to the check before publication | passed |

Each check reported no findings, no exceptions and nothing uninspected.

## Quick start card update (2026-11-05)

Issue #466 / PR #494 updates only `README.md` in each public repository to acquire models from the Mac sidebar headings. The optional CLI instructions remain separate. The owner specified full-payload local inspection before the update and anonymous hub inspection afterwards; this supersedes the earlier pre-update equality-gate requirement for this card change. The canonical scanner was not modified. The [public summary](mac-models-distribution-20261104/card-update-20261105.json) records the checked payload manifests and remote revisions; full reports remain private.

| Model | Updated `main` | Local payload | Local result (JST) | Anonymous history result (JST) |
|---|---|---|---|---|
| Qwen3.8 27B | `3d50b382595c171c14ec03d56f98b16d5da949a8` | 12 files, 16,852,544,966 bytes | passed, 2026-11-05 01:26 | passed, 2026-11-05 02:11; 6 commits, 17 unique objects |
| Clef-Flash 9B | `873722e26cc47bd442af8b8d6103599acbdd6a09` | 25 files, 7,278,592,778 bytes | passed, 2026-11-05 01:05 | passed, 2026-11-05 01:31; 19 commits, 31 unique objects |

Both checks used scanner `63501ba4f0d44b8a0f05127025407875f7cbd983dc573919d6905b816d9f7baf` and completed with zero unresolved findings and zero errors. Each local payload contained every distribution file, including the updated card. The anonymous check included all refs, pull-request refs and complete history, and its head manifest equaled the inspected local payload. Both repositories remain public. All non-card files retained their previous object IDs.

The application catalog and download examples still pin Qwen `8e24aebe036eae577d4a3bd019ba728a73358318` (`v1.0.0`) and Clef `12e30be4e391310377c534524df8962359a08b0f`. This card update does not change those inventories. Actual anonymous acquisition through the application library is recorded in [the F validation](mac-language-models-20261104.md); physical-device and ordinary UI acceptance remain separate.
