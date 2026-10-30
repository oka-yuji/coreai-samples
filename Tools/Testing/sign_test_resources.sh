#!/bin/bash
set -euo pipefail

if [[ "${CODE_SIGNING_ALLOWED:-NO}" != "YES" ]]; then
    exit 0
fi

: "${EXPANDED_CODE_SIGN_IDENTITY:?Missing test signing identity}"
: "${CODESIGNING_FOLDER_PATH:?Missing test bundle path}"

shopt -s nullglob
for resource_bundle in "$CODESIGNING_FOLDER_PATH"/*.bundle; do
    if /usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$resource_bundle/Info.plist" >/dev/null 2>&1; then
        printf 'Expected a resource-only test bundle: %s\n' "$resource_bundle" >&2
        exit 1
    fi
    /usr/bin/codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" --timestamp=none "$resource_bundle"
done
