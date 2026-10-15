# Conversion maintenance

The conversion scripts are repository maintenance tools. They are optional for building
or running the sample app and do not define a stable external API. Instructions are in
[Tools/Conversion](../Tools/Conversion/README.md).

## Ownership and local data

New Core AI conversion work is maintained in this repository. The previous private lab
is retained as historical evidence and is no longer a destination for new work. Its
history, raw experiment records and model weights were not imported.

Tracked content consists of the runner, model-specific recipes, tests, instructions and
reviewed numeric summaries. .local/conversion holds personal paths, the pinned public
exporter checkout, its Python/Swift build products, complete reports and model outputs.
It is ignored by Git. These local files are not backed up by a Git push. Downloadable
tools and model inputs are reconstructible; preserve any irreplaceable local experiments
in a private backup before deleting their workspace.

The shared model-job lock lives outside the repository, under the maintainer's cache
directory. All concurrent worktrees on one machine must agree on that lock. Setup takes
the lock too, so it cannot replace an environment while a cooperating model job uses it.
Legacy jobs must finish before switching their lock location.

Optional GLiNER fixture checks accept GLINER_FIXTURES_DIR and no longer search the old
lab by default. When moving a reference set, retain its models/coreai/gliner and
docs/results layout; the tests use relative references between them. Those private
reference files do not become repository test assets.

## Relocation validation, 2026-10-15

The fixed Qwen3-0.6B recipe was run on an M4 Max with 128 GB, macOS 27.0 (26A428),
Xcode 27.0 (27A266a). Inputs were the existing HF snapshot at
c1899de289a04d12100db370d81485cdf75e47ca and a new public apple/coreai-models checkout
at 5de95c6cd29e5e390fb36d87e13b2396668f7974. The exporter environment was installed
from its frozen uv lock, and llm-runner was built from its pinned Swift dependencies.
The old virtual environment and build directory were not copied or linked.

The model configuration remained FP16, no compression, macOS, context 1024. The
reference was FP32 on the CPU. Criteria remained maximum probability error 0.03 and
zero top-1 differences except matching top-two candidates within a 0.01 reference margin.
Both runs denied file reads from the old lab directory at the OS sandbox boundary.
Existing jobs' lock protocol was respected during the transition.

| Run | Status | Seconds | Reference / export / verification |
|---|---|---:|---|
| [Fresh](results/conversion-maintenance/fresh.json) | Passed | 160.411 | Executed / executed / executed |
| [Repeat](results/conversion-maintenance/reuse.json) | Passed | 5.164 | Reused / reused / reused |

All 44 measured token positions had the same top-1 token, without near-tie exemptions.
The maximum probability difference was 0.00408485897. These timings describe this
workspace and run; other build activity was present, so they are not a controlled
comparison with earlier timings or a claim about export speed on other machines.
Reused stages remain labelled reused, not fresh measurements.

The imported runner and model adapters retained their prior logic; relocation changes
their module paths and documentation. The setup and summary entry points have separate
boundary tests. Independent review found and corrected inherited uv environment overrides,
external build-directory symlinks, and a manual bindings example outside the ignored
workspace. Setup now filters inherited overrides, pins its target environment, rejects
escaping paths, and refuses modified exporter checkouts without resetting them.

All 54 conversion tests passed in the dedicated Python environment, including the final
setup guards. Setup was also rerun successfully after those guards were added.
The sample's Swift tests completed successfully (284 tests reported across fourteen
test executables), with model-enabled and tokenizer-dependent checks left opt-in.
The required format check and Release build passed. Fifteen
external GLiNER reference files were copied byte-for-byte to the independent private
reference cache, and the affected fixture suites passed there. Model-enabled app tests
were not enabled: this change does not alter the app's inference runtime or UI.

SwiftLint passed with the existing rules and baseline. The temporary macOS worktree
used two equivalent path spellings, causing the baseline's relative paths not to match.
All 35 reported diagnostics matched the existing baseline's file, rule and source text;
normalizing only its path prefix in a temporary copy yielded zero violations. The
repository baseline was not changed, and the unmodified main checkout passed its normal
lint command. This is a temporary validation-path issue, not 35 new source violations.

## Limits and next work

This is a bounded conversion smoke check, not a model release qualification. Swift
receives raw reference token IDs. Long context, tokenizer equivalence, endurance,
iPhone deployment and quantization quality are outside this recipe. Existing Clef,
GLiNER, speech and other architectures need explicit adapters and their own criteria.

Future work extends these maintained recipes only when needed for a conversion task.
It must not import fixed historical output paths or detached background processes into
the runner. The current request adds no new model, model release or demo feature.
