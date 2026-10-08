#!/usr/bin/env bash
set -euo pipefail
check_stage=setup
trap 'check_code=$?; if [[ "$check_code" -ne 0 ]]; then printf "CHECK FAIL: %s\n" "$check_stage" >&2; exit 1; fi' EXIT
check_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$check_root"
check_models=${COREAI_TEST_BUNDLES-}
for check_variable in ${!COREAI_@} ${!GLINER_@} ${!DECIDE_@} ${!CLEF_@}; do
    unset "$check_variable"
done
check_stage=model-inputs
if [[ -n "$check_models" ]]; then
    case "$check_models" in :*|*:|*::* ) false ;; esac
    IFS=: read -r -a check_bundles <<< "$check_models"
    for check_bundle in "${check_bundles[@]}"; do
        if [[ ! -d "$check_bundle" || ! -f "$check_bundle/metadata.json" || ! -f "$check_bundle/tokenizer/tokenizer.json" ]]; then
            printf 'Missing bundle folder, metadata.json or tokenizer/tokenizer.json: %s\n' "$check_bundle" >&2
            false
        fi
    done
    command -v python3 >/dev/null
fi
check_stage=build-kit
printf 'CHECK RUN: %s\n' "$check_stage"
swift build --package-path CoreAISamplesKit
check_stage=test-without-models
printf 'CHECK RUN: %s\n' "$check_stage"
swift test --package-path CoreAISamplesKit --no-parallel
check_stage=build-verify
printf 'CHECK RUN: %s\n' "$check_stage"
swift build --package-path Tools/Verify -c release
check_stage=build-app
printf 'CHECK RUN: %s\n' "$check_stage"
check_output="$check_root/.build/check"
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release \
    -destination 'platform=macOS' -derivedDataPath "$check_output/DerivedData" build
check_stage=agent-offline
printf 'CHECK RUN: %s\n' "$check_stage"
"$check_output/DerivedData/Build/Products/Release/CoreAISamples.app/Contents/MacOS/CoreAISamples" --agent-selftest --offline
validate_records() {
    python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path
mode, path = sys.argv[1:]
rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
if not rows or any(not isinstance(row, dict) or row.get("error") is not None for row in rows):
    raise SystemExit("Missing records or an error was reported.")
if mode == "compare":
    records = [row for row in rows if row.get("record") == "compare"]
    expected = {(prompt, prefill) for prompt in ("en", "code") for prefill in ("prefill-function", "main-only")}
    if len(records) != 4 or {(row.get("prompt"), row.get("prefill")) for row in records} != expected:
        raise SystemExit("Expected both prompts in both prefill modes.")
    for row in records:
        if row["prefill"] == "prefill-function":
            tokens = row.get("tokens")
            if not isinstance(tokens, list) or not tokens or row.get("match") is not True or row.get("official_tokens") != tokens or row.get("identical_steps") != len(tokens) or row.get("max_abs_diff") != 0 or row.get("first_mismatch") is not None:
                raise SystemExit("The matching prefill path differs: " + row["prompt"])
    print("CHECK RESULT: both prompts match at every step on the prefill-function path; main-only records are diagnostic.")
else:
    records = [row for row in rows if row.get("record") == "state-check"]
    turns = [row for row in rows if row.get("record") == "turn"]
    if len(turns) != 2 or {row.get("turn") for row in turns} != {1, 2} or any(row.get("runtime") != "own" or row.get("output_tokens", 0) <= 0 for row in turns):
        raise SystemExit("Expected two completed own-runtime turns.")
    if len(records) != 1 or records[0].get("pass") is not True or records[0].get("tokens", 0) <= 0:
        raise SystemExit("The processed-state check failed or is missing.")
    print("CHECK RESULT: the processed-state check passed after two turns.")
PY
}
if [[ -n "$check_models" ]]; then
    check_stage=test-with-models
    printf 'CHECK RUN: %s\n' "$check_stage"
    COREAI_TEST_BUNDLES="$check_models" swift test --package-path CoreAISamplesKit -c release --no-parallel
    check_stage=compare
    printf 'CHECK RUN: %s\n' "$check_stage"
    Tools/Verify/.build/release/coreai-verify compare --model "${check_bundles[0]}" --max-tokens 64 \
        --prompt 'en=List three fruits, one per line.' \
        --prompt 'code=Write a Python function that adds two integers.' | tee "$check_output/compare.jsonl"
    validate_records compare "$check_output/compare.jsonl"
    check_stage=state-check
    printf 'CHECK RUN: %s\n' "$check_stage"
    Tools/Verify/.build/release/coreai-verify chat --model "${check_bundles[0]}" --runtime own --reasoning off --check-state --max-tokens 32 \
        --prompt 'The code word is amber. Reply with one word.' --prompt 'Repeat the code word.' | tee "$check_output/state.jsonl"
    validate_records state "$check_output/state.jsonl"
fi
printf 'CHECK PASS\n'
