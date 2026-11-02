import argparse
import json
import re
import subprocess
import sys

pattern = re.compile(
    rb"[/]Users[/](?!runner[/]work[/])[^\s/`'\"]+"
    rb"|[-]Users[-][^\s/`'\"-]+[-]"
    rb"|[/]home[/][^\s/`'\"]+"
    rb"|[/]Volumes[/][^\s/`'\"]+"
    rb"|[/]var[/]folders[/]"
)
escaped = re.compile(rb"\\/|\\u002[fF]|%2[fF]")
kinds = ["issue", "pr", "comment", "review", "review-comment"]
query = "{ nodes(ids: %s) { ... on Comment { userContentEdits(first: 100) { pageInfo { hasNextPage } nodes { diff deletedAt } } } } }"


def lines(data):
    data = escaped.sub(b"/", data)
    return sorted({data.count(b"\n", 0, match.start()) + 1 for match in pattern.finditer(data)})


def fail():
    print("Could not read all issues, pull requests, comments, reviews and edits with gh.", file=sys.stderr)
    sys.exit(2)


def gh(*args, data=None):
    return json.loads(subprocess.run(["gh", "api", *args], input=data, stdout=subprocess.PIPE, check=True).stdout)


def listing(path):
    return [item for page in gh("--paginate", "--slurp", f"repos/{repository}/{path}") for item in page]


def scan():
    entries = []
    for issue in listing("issues?state=all&per_page=100"):
        kind = "pr" if "pull_request" in issue else "issue"
        entries.append((kind, issue["number"], issue["node_id"], [("title", issue["title"]), ("body", issue["body"])]))
        if kind == "pr":
            for review in listing(f"pulls/{issue['number']}/reviews?per_page=100"):
                entries.append(("review", review["id"], review["node_id"], [("", review["body"])]))
    for kind, path in (("comment", "issues/comments"), ("review-comment", "pulls/comments")):
        for comment in listing(f"{path}?per_page=100"):
            entries.append((kind, comment["id"], comment["node_id"], [("", comment["body"])]))
    found = []
    for start in range(0, len(entries), 100):
        batch = entries[start:start + 100]
        request = json.dumps({"query": query % json.dumps([entry[2] for entry in batch])}).encode()
        nodes = gh("graphql", "--input", "-", data=request)["data"]["nodes"]
        if len(nodes) != len(batch):
            fail()
        for (kind, number, _, texts), node in zip(batch, nodes):
            edits = node["userContentEdits"]
            if edits["pageInfo"]["hasNextPage"]:
                fail()
            versions = [(part, 0, text) for part, text in texts]
            versions += [(texts[-1][0], index, edit["diff"]) for index, edit in enumerate(edits["nodes"], 1) if edit["deletedAt"] is None]
            for part, version, text in versions:
                place = f"{kind} #{number} {part}" if part else f"{kind} {number}"
                suffix = f" (edit {version})" if version else ""
                found += [((kinds.index(kind), number, part != "title", version, line), f"{place}:{line}{suffix}") for line in lines((text or "").encode())]
    return [item for _, item in sorted(found)]


parser = argparse.ArgumentParser()
parser.add_argument("repository")
repository = parser.parse_args().repository
try:
    found = scan()
except (OSError, subprocess.CalledProcessError, LookupError, TypeError, ValueError, AttributeError):
    fail()
for item in found:
    print(item)
if found:
    print(f"{len(found)} lines contain an absolute local path.", file=sys.stderr)
sys.exit(1 if found else 0)
