import argparse
import json
from pathlib import Path
import signal
import sys

from .recipe import expand, load_recipe, preflight, source_paths
from .runner import run
from .storage import PipelineError, read_json


def main():
    parser = argparse.ArgumentParser(description="Run model-specific recipes with verified stage reuse.")
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--bindings", type=Path, help="JSON object of local path variables")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--lock-dir", type=Path, help="Shared machine lock for model jobs")
    parser.add_argument("--refresh", action="store_true", help="Re-execute stages without accepting cached results")
    args = parser.parse_args()
    try:
        bindings = read_json(args.bindings) if args.bindings else {}
        if not isinstance(bindings, dict):
            raise PipelineError("bindings must be a JSON object")
        for item in args.set:
            if "=" not in item:
                raise PipelineError("--set requires NAME=VALUE")
            name, value = item.split("=", 1)
            bindings[name] = value
        repo = Path(__file__).resolve().parents[2]
        recipe = load_recipe(args.recipe, bindings, repo)
        preflight(recipe)
        if args.action == "plan":
            stages = []
            for stage in recipe["steps"]:
                values = {**recipe["variables"], **{d: f"<artifact:{d}>" for d in stage["depends"]},
                          "out": "<new-stage-output>", "scratch": "<temporary>"}
                stages.append({**expand(stage, values),
                               "resolved_inputs": [str(p) for p in source_paths(stage, recipe["variables"])]})
            print(json.dumps({"recipe": recipe["name"], "stages": stages}, indent=2))
            return 0
        if args.work_dir is None:
            raise PipelineError("run requires --work-dir")

        def stop(_signum, _frame):
            raise KeyboardInterrupt("Termination requested")

        signal.signal(signal.SIGTERM, stop)
        run(recipe, args.work_dir, lock_dir=args.lock_dir, refresh=args.refresh)
        return 0
    except KeyboardInterrupt:
        print("Pipeline cancelled; incomplete results will not be reused.", file=sys.stderr)
        return 130
    except (PipelineError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"Pipeline error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
