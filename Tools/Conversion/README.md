# Conversion maintenance

This is the repository maintainer's conversion workspace, not a separately supported
package or a stable public API. It is optional: the sample app does not depend on Python
or this tooling. Model-specific conversion and reference checks share one entry point. Successful
stages can be reused after their inputs, environment and outputs have been checked again.
The pipeline does not choose a model's precision or relax its verification criteria.

Python 3.11+ is sufficient for the orchestrator and model-free tests. Each recipe can
select a separate Python environment for its model commands.

~~~sh
python3 -m Tools.Conversion plan path/to/recipe.json --bindings .local/conversion/local-paths.json
python3 -m Tools.Conversion run path/to/recipe.json --bindings .local/conversion/local-paths.json \
  --work-dir .local/conversion/work/example \
  --lock-dir "$HOME/Library/Caches/coreai-samples/heavy-job"
python3 -m unittest discover -s Tools/Conversion/tests -p test_pipeline.py -v
~~~

Run from the repository root. The plan command validates local inputs and executables
and prints commands without importing the model environment or writing artifacts.
Run adapter tests in the model environment as described by the recipe. Numerical
adapter tests skipped because NumPy is missing are not a complete validation result.
The run command creates a report under work-dir/runs and immutable completed objects
under work-dir/objects. Each execution has a distinct object path; an atomic index
selects the current reusable object without changing past report links. One run owns
a work directory at a time, so refreshed dependencies cannot move underneath a reader.
Reports contain the source run for reused stages. A reused
measurement is never presented as a fresh measurement.

A pipeline containing a heavy stage holds the shared machine lock from host/input
fingerprinting through execution, output checks and cached reuse. Another owner's
lock is never removed. If a child process or its process group cannot be confirmed
stopped, the report keeps the primary failure and cleanup details separately, and
the work, stage and machine locks remain in place. Confirm that recorded process
group and child have stopped before recovering only that failed run's locks.
Zombie-only groups are treated as stopped after observation; permission errors,
failed observations or remaining live members are not silently accepted.


The [Qwen3-TTS CustomVoice recipe](recipes/qwen3_tts/README.md) uses a separate
Transformers 4.57.3 environment to export FP32 neural components and compare them
with the original Japanese speech path. Use generic plan/run with local bindings.

The [Qwen3-TTS 1.7B CustomVoice recipe](recipes/qwen3_tts_17b/README.md) extends the
shared TTS adapter with a separately pinned source and output directory.

## Local workspace

Run from the sample repository root. Use Python 3.11+, Git, uv 0.9+, Xcode 27 with
Metal Toolchain, and macOS 27 on Apple Silicon. Fetch the input snapshot once:

~~~sh
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
python3 -m Tools.Conversion.local setup
python3 -m Tools.Conversion.local plan
python3 -m Tools.Conversion.local run
~~~

Setup clones Apple's public converter at the recipe's fixed revision, installs its
frozen Python dependencies, and builds its pinned Swift llm-runner. It refuses an
existing checkout at a different revision or with tracked edits. It does not overwrite
or clean such a checkout. Python/build directories and their ancestors must remain in
the chosen workspace. Setup filters inherited environment overrides and pins uv to its
dedicated .venv, and takes the same machine lock as model jobs. An interrupted initial setup can require inspecting that
checkout before retrying. Models are never downloaded by setup. Use --snapshot for
an existing HF snapshots/<revision> directory, and --workspace to select another local
workspace. The default is .local/conversion; keep any alternative outside Git as well.

The local paths JSON is recipe-specific. Repeat run to resume/reuse; --refresh forces
new stages. --recipe selects another maintained directory under recipes. New architectures
need their own adapter and criteria. The Qwen recipe is only an FP16 smoke check.

Model jobs share ~/Library/Caches/coreai-samples/heavy-job across worktrees. If migrating
from a previous lock location, wait for its owners to finish and coordinate all processes
on the new location. Never delete an active lock. COREAI_HEAVY_JOB_LOCK or --lock-dir can
select an agreed alternative; all model processes must use the same location.

Local paths, cloned tools, virtual environments, complete reports and cache generations
stay under the ignored .local directory. Cache generations are retained until deliberately
retired. They are not a public dataset. Reports and logs can contain environment values,
commands, paths and fixture text. For a shareable numeric projection:

~~~sh
python3 -m Tools.Conversion.summary .local/conversion/work/qwen3_06b/runs/<run>/report.json
~~~

Optionally pass --comparison with that run's comparison.json to include aggregate token
metrics. Review the result before sharing. The projection omits arbitrary text, paths,
commands, environment variables and prompts. It preserves failed/incomplete/reused states.
Keep model identity, precision, revisions, criteria and limitations alongside any published
numbers. It does not convert a smoke result into a production qualification.

The tools are this repository's MIT-licensed code. Apple's converter is installed separately
under BSD-3-Clause. Qwen3-0.6B has Apache-2.0 terms. Any redistributed upstream code or model
assets keep their own required notices; inspect each new model's terms independently.

## Recipe version 1

Recipes are trusted local executable configuration, not an untrusted data format.
Commands run without a shell, using an argument array. This is an orchestration tool,
not a sandbox for arbitrary commands.
Executables use absolute paths or names resolved through PATH. A relative path such
as ./convert is rejected; execution uses the same absolute executable that was hashed.

~~~json
{
  "version": 1,
  "name": "example",
  "variables": {
    "python": "/path/to/environment/bin/python",
    "snapshot": "/path/to/pinned/snapshot"
  },
  "python": "$python",
  "environment": {
    "HF_HUB_OFFLINE": "1"
  },
  "steps": [
    {
      "id": "reference",
      "role": "reference",
      "command": ["$python", "$repo/reference.py", "$snapshot", "$out"],
      "inputs": ["$repo/reference.py", "$snapshot"],
      "outputs": ["reference.json"],
      "heavy": true,
      "timeout_s": 3600,
      "min_free_bytes": 10000000000
    },
    {
      "id": "verify",
      "role": "verify",
      "depends": ["reference"],
      "command": ["$python", "$repo/compare.py", "$reference", "$out"],
      "inputs": ["$repo/compare.py"],
      "outputs": ["comparison.json"],
      "checks": [
        {"file": "comparison.json", "key": ["passed"], "equals": true}
      ]
    }
  ]
}
~~~

This is a schema example; reference.py and compare.py represent a model-specific
implementation and are not supplied by the orchestrator.

- Variables use Python string.Template syntax. Literal dollar signs use $$.
  Local JSON bindings or --set NAME=VALUE override defaults. Variables can reference
  earlier-resolvable variables; unresolved references and cycles fail.
- Built-ins: repo, recipe_dir, out (new artifact directory), scratch (temporary
  workspace). A declared dependency's ID resolves to its completed artifact directory.
- Stages are ordered; dependencies must name earlier stages. Independent reference
  and export stages should not depend on each other. A fixture change can then
  invalidate reference/verification while leaving export reusable.
- inputs lists exact source paths or glob patterns. Every pattern must match.
  Include the complete source dependency closure: imported helpers, fixtures,
  tokenizer/config/weights, compression configuration and editable upstream code.
  Directory inputs include all files except Git internals and Python bytecode.
  Regular file symlinks such as HF snapshot entries are hashed by content.
  Directory symlinks are rejected to avoid silently incomplete inventories.
- outputs declares required relative files/directories. The complete artifact directory
  is hashed as well, so files consumed by downstream commands cannot bypass integrity
  checks merely because they were omitted from the required-output list. Empty, missing or escaping
  outputs are rejected. Output symlinks are rejected. A successful exit code by itself
  is insufficient to complete a stage.
- verify stages require explicit checks. A check traverses a JSON key/index array and
  compares exact JSON type/value with equals, or finite numbers with min/max.
  Criteria are part of the stage fingerprint. Missing fields and non-finite values fail.
- benchmark stages always execute. Other stages cache by default; cache:false disables
  reuse. --refresh executes all stages again without deleting their previous versions.
- heavy:true uses the shared machine mkdir lock. A busy lock fails clearly; it is never
  stolen or automatically treated as stale. Inspect the owner and retry after its job
  finishes. This also applies to CPU reference jobs that load model weights.
- timeout_s bounds each process. The runner owns its process group and terminates
  descendants on failure, timeout or cancellation. Logs and incomplete outputs remain
  under that run for diagnosis, without a completed cache manifest.
  Stage commands must stay in that process group: daemonization, setsid,
  start_new_session in descendants, and existing scripts' --detach modes are not
  supported. Invoke the foreground forms. This contract is necessary for portable
  process-group cleanup; the tool is not a machine-wide process supervisor.
- min_free_bytes is a per-stage admission check. A model-specific memory/disk monitor
  can remain inside the stage command; this option is not continuous resource monitoring.

## Reuse contract

An object is reusable only when its completed manifest matches the current fingerprint,
every required output still hashes identically, and the declared checks still pass.
Fingerprints include the stage configuration, source bytes, dependency output
inventories, executable, pipeline code and host environment. A configured Python
environment is probed for its version and installed distributions; installed library
contents are also hashed, including same-version local patches. Editable upstream
source outside site-packages must be declared as an input.

The child environment only inherits PATH, HOME, TMPDIR, LANG, LC_ALL and DEVELOPER_DIR;
PYTHONNOUSERSITE and PYTHONDONTWRITEBYTECODE are set. Other required variables belong
in the recipe's environment mapping. Do not put secrets into a recipe: the environment
is recorded for provenance. This workflow is intended to use already-downloaded inputs.

Input and toolchain identities are checked again after execution. Each new attempt has
its own output directory. The runner writes a completed manifest only after the process,
outputs, checks and identity checks succeed, then moves the object into the cache.
Previous objects are retained at their original paths. A corrupt object gets a
quarantine record pointing to it; refresh and benchmark runs create new generations.
Successful scratch directories are removed before publication.
No source model, previous experiment, global HF reference or published bundle is
intentionally rewritten by the runner.

Cache reuse is only as complete as the declared dependency boundary. External commands
must not read undeclared changing files, clock/random state without fixed seeds,
unrecorded services or hidden working-directory configuration. Declare those inputs
or disable caching for that stage. A recipe's successful checks describe their stated
scope; they do not certify the model against a model's full production checks.

## Failure and maintenance

On failure, inspect the run's report.json, report.md and stage/process.log. Fix the
cause and run the same command; completed predecessors remain eligible for reuse.
Do not create manifests manually or import old experiments as successful cache entries.
Each recipe change is version-controlled and should be reviewed with its checks.

Cache, diagnostic attempts and quarantine have no automatic eviction. Inspect disk
usage and remove only run/cache directories that you deliberately retire. Never use
this cache folder as an input directory: overlapping inputs and work directories fail.

The maintenance contract and relocation checks are tracked in issue #118.

The [Image Research 512 recipe](recipes/qwen_image21/README.md) converts
Qwen-Image-2.1-Turbo into a fixed FP32 macOS image generation and editing profile.
Select `--recipe qwen_image21` for its separate frozen environment and source checks.
Its model assets retain the Qwen Research License's non-commercial research and
evaluation restrictions. Mobile rewriting is investigated separately from Mac validation.

The [pinned Qwen3-0.6B recipe](recipes/qwen3_06b/README.md) connects source-revision
checks, the FP32 reference, an official FP16 export and Swift numerical checks.
Its [relocation record](../../docs/conversion-maintenance.md) records the checks and their limits. It is a bounded
integration example and does not replace a model's production qualification.

The [EmbeddingGemma 2 recipe](recipes/embeddinggemma2/README.md) uses a separate frozen environment for text/vision conversion, source and native-build stamps, and twenty raw-input fidelity checks. Select it with `--recipe embeddinggemma2` on setup/plan/run. Its export stages deliberately depend on the FP32 reference used for wrapper verification.

The [Magpie TTS v2607 recipe](recipes/magpie/README.md) defines an FP32 macOS research
profile with four Core AI components, pinned NeMo references and Japanese audio
checks. It uses the generic plan/run commands with explicit local bindings.

The [TeleOCR recipe](recipes/teleocr/README.md) exports the pinned TeleOCR vision encoder, token
embedding and decoder as macOS Core AI assets with a separate Transformers 4.57.3 environment. Select it
with `--recipe teleocr`. Its Python checks cover the stateless parts; the stateful decoder check is deferred to #310 (Swift).
