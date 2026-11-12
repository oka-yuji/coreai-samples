import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


URL = "https://data.mendeley.com/public-files/datasets/njdkntcpc9/files/08bb4f43-6dfa-4995-b268-42fa0690ba6b/file_downloaded"
BYTE_COUNT = 888985
SHA256 = "0dfd4a5bf801bf79ddb5fb1f6bc8d023487c36dd6a2a70b312cdfa9fa6568d83"
LIMIT = 2 * 1024 * 1024


def validate(data):
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != BYTE_COUNT or digest != SHA256:
        raise ValueError("The bytes do not match the pinned Mendeley v1 source")
    return {"bytes": len(data), "sha256": digest, "source": URL}


def acquire(output, opener=urllib.request.urlopen):
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError("Refusing an existing output")
    with opener(URL, timeout=60) as response:
        if response.status != 200:
            raise ValueError("The provider did not return HTTP 200")
        data = response.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError("The response exceeds the 2 MiB limit")
    summary = validate(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".sales-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Explicitly acquire or check noncommercial Mendeley sales data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.check:
        with args.check.open("rb") as handle:
            summary = validate(handle.read(LIMIT + 1))
    else:
        summary = acquire(args.output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
