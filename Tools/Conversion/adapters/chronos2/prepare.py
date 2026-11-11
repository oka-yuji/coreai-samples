import argparse
from pathlib import Path

from .artifacts import input_identity, load_contract, load_fixtures, new_output, read_artifact, require_equal
from .artifacts import validate_source, write_new
from .setup import check_setup


def prepare(model, fixtures, snapshot, setup_stamp, out):
    out = new_output(out)
    contract = load_contract(model)
    load_fixtures(fixtures)
    validate_source(snapshot, contract)
    setup = check_setup(Path(model).parent, setup_stamp)
    ready = {"ready": True, "identity": input_identity(model, fixtures),
             "source_files": contract["files"], "setup": setup}
    write_new(out / "ready.json", ready)
    return ready


def check_ready(path, model, fixtures, snapshot=None):
    contract = load_contract(model)
    cases = load_fixtures(fixtures)
    identity = input_identity(model, fixtures)
    ready = read_artifact(path)
    expected = {"ready": True, "identity": identity, "source_files": contract["files"]}
    if not isinstance(ready, dict) or set(ready) != {*expected, "setup"}:
        raise ValueError("Invalid preparation record")
    require_equal({key: ready[key] for key in expected}, expected, "preparation record")
    if snapshot is not None:
        validate_source(snapshot, contract)
    return contract, cases, identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--setup-stamp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.fixtures, args.snapshot, args.setup_stamp, args.out)


if __name__ == "__main__":
    main()
