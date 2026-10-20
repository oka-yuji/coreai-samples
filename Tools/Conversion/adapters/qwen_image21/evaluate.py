import argparse
import json
from pathlib import Path

from .evaluation import aggregate_runs, live_memory_peak, quality_summary, validate_inventory, validate_profile
from .fixtures import save_json
from .profile import baseline_profile


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("baseline-profile")
    profile.add_argument("--bundle", type=Path, required=True)
    profile.add_argument("--dependency-lock", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    for name in ("profile", "aggregate", "memory", "quality"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
        command.add_argument("--output", type=Path)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("root", type=Path)
    inventory.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "baseline-profile":
        result = baseline_profile(args.bundle, args.dependency_lock)
        validate_profile(result)
    elif args.command == "inventory":
        result = validate_inventory(args.root, json.loads(args.manifest.read_text())["inventory"])
    else:
        value = json.loads(args.input.read_text())
        if args.command == "profile":
            result = {"profile_sha256": validate_profile(value)}
        elif args.command == "aggregate":
            result = aggregate_runs(value["runs"], value.get("minimum_samples", 20))
        elif args.command == "memory":
            result = live_memory_peak(value["allocations"], value.get("unknown_categories", []))
        else:
            result = quality_summary(value["reference"], value["candidate"], value["required_axes"])
    if getattr(args, "output", None) is not None:
        if args.output.exists():
            raise FileExistsError("Evaluation output must not overwrite an earlier record")
        save_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
