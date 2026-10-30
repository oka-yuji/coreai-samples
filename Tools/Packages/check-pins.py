import argparse
import json
import plistlib
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
kit_lock = root / "CoreAISamplesKit/Package.resolved"
verify_lock = root / "Tools/Verify/Package.resolved"


def pins(lock):
    return {pin["identity"]: pin["state"] for pin in json.loads(lock.read_text())["pins"]}


def selected(state):
    return {
        dependency["packageRef"]["identity"]: dependency["state"].get("checkoutState", dependency["state"])
        for dependency in json.loads(state.read_text())["object"]["dependencies"]
        if dependency["packageRef"]["kind"] != "fileSystem"
    }


def value(states, identity, absent="no pin"):
    state = states.get(identity)
    return absent if state is None else f"version {state.get('version')} revision {state.get('revision')}"


def default_derived_data():
    base = Path.home() / "Library/Developer/Xcode/DerivedData"
    project = (root / "CoreAISamples.xcodeproj").resolve()
    found = [
        info.parent
        for info in sorted(base.glob("*/info.plist"))
        if Path(plistlib.loads(info.read_bytes()).get("WorkspacePath", "/")).resolve() == project
    ]
    if len(found) != 1:
        listed = "".join(f"\n  {folder}" for folder in found)
        sys.exit(
            f"FAIL: expected one folder in {base} whose info.plist has WorkspacePath {project}, found {len(found)}.{listed}\n"
            "Fix: pass the DerivedData folder of the build as an argument, after building with xcodebuild if there is none."
        )
    return found[0]


parser = argparse.ArgumentParser(description="Fail when the packages that SwiftPM or Xcode selected differ from Package.resolved.")
parser.add_argument(
    "derived_data",
    nargs="?",
    type=Path,
    help="DerivedData folder of the xcodebuild run (default: the folder of this checkout in ~/Library/Developer/Xcode/DerivedData)",
)
parser.add_argument("--with-verify", action="store_true", help="also check Tools/Verify/.build against Tools/Verify/Package.resolved")
arguments = parser.parse_args()
derived_data = arguments.derived_data or default_derived_data()
checks = [
    (
        root / "CoreAISamplesKit/.build/workspace-state.json",
        kit_lock,
        "Then run swift build or swift test for CoreAISamplesKit again with --force-resolved-versions.",
    ),
    (
        derived_data / "SourcePackages/workspace-state.json",
        kit_lock,
        f"Then delete {derived_data / 'SourcePackages'} and run xcodebuild again with -onlyUsePackageVersionsFromResolvedFile.",
    ),
]
if arguments.with_verify:
    checks.append(
        (
            root / "Tools/Verify/.build/workspace-state.json",
            verify_lock,
            "Then run swift build for Tools/Verify again with --force-resolved-versions.",
        )
    )
kit, verify = pins(kit_lock), pins(verify_lock)
problems = [
    f"{verify_lock}: {identity}: {value(verify, identity)}, but {kit_lock} has {value(kit, identity)}"
    for identity in sorted(kit)
    if value(verify, identity) != value(kit, identity)
]
fixes = [f"Then give {verify_lock} the same pins as {kit_lock}."] if problems else []
for state, lock, fix in checks:
    if not state.is_file():
        problems.append(f"{state}: not found; build first (for Xcode, pass the DerivedData folder that xcodebuild used)")
        continue
    recorded, chosen = pins(lock), selected(state)
    found = [
        f"{state}: {identity}: selected {value(chosen, identity, 'nothing')}, but {lock} has {value(recorded, identity)}"
        for identity in sorted(recorded.keys() | chosen.keys())
        if value(chosen, identity, "nothing") != value(recorded, identity)
    ]
    problems += found
    if found:
        fixes.append(fix)
    else:
        print(f"PASS {state}: {len(chosen)} packages match {lock}")
if problems:
    sys.exit(
        "\n".join(
            [
                "FAIL: the packages that SwiftPM or Xcode selected differ from Package.resolved.",
                *problems,
                "Fix: Package.resolved is the source of truth. If a pin there is missing, extra or wrong, correct it first.",
                *fixes,
            ]
        )
    )
print(f"PASS {verify_lock}: the {len(kit)} pins of {kit_lock} match")
