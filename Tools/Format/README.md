# Swift format checks

The checker uses the repository's `.swift-format`, Python 3.9+, Git, and
`xcrun swift-format` from the selected Xcode installation. No personal
configuration directory or additional Python packages are required.

Run from the repository root:

```sh
git fetch origin main
python3 Tools/Format/check_format.py --base origin/main
```

`--base` checks Swift files changed from the merge-base of the given ref and
`HEAD` to the working tree, including committed, staged, and unstaged changes.
It also checks untracked Swift files that Git does not ignore.

For an audit of all tracked and non-ignored untracked Swift files:

```sh
python3 Tools/Format/check_format.py --all
```

Both modes skip deleted files, reject Swift symlinks, and reject files with
both staged and unstaged changes. They lint the working-tree contents without
rewriting them. An audit can expose existing formatting issues outside a PR's
changes; the PR check uses `--base`.

| Exit code | Meaning |
|---|---|
| `0` | Selected files passed, no Swift files were selected, or only the fixed wire-name exception applied. |
| `1` | A formatter invocation failed or a selected Swift file was a symlink. |
| `2` | Invalid arguments, missing configuration, Git or tool launch failure, or partial staging prevented the check. |

`check_swift_format.py` selects files and invokes the formatter.
`check_format.py` is the public entrypoint and preserves the single
`AlwaysUseLowerCamelCase` exception for `WebSearchArguments.max_results` in
`CoreAISamples/Demos/Agent/AgentPromptBuilder.swift`. It requires the exact
`@Generable` type and `var max_results: Int?` declaration, with no duplicate
declaration or additional diagnostic. This keeps the external tool schema's
wire name stable.

Run the regression suite with:

```sh
python3 -m unittest discover -s Tools/Format/tests -v
```

The repository's `./check.sh` also runs this suite.

The suite uses temporary Git repositories and a controlled formatter to check
file selection, failure handling, and the naming exception. It also uses the
real `xcrun swift-format` to verify valid and invalid fixtures without modifying
them. That integration test is skipped when `xcrun` is unavailable; a skipped
test does not verify the installed formatter.
