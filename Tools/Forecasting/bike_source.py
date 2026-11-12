import argparse
import csv
from datetime import datetime, timedelta
import hashlib
from http.client import HTTPException
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import urllib.request
import zipfile


MAX_INPUT_BYTES = 2 * 1024 * 1024
CSV_NAME = "SeoulBikeData.csv"
SOURCE_BYTES = 604166
SOURCE_SHA256 = "373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60"
NORMALIZED_BYTES = 604168
NORMALIZED_SHA256 = "d2f7a0f0ba95e82678ead794429a18a7f696aa3b82c9b9ec34ec47806ff0767b"
ZIP_BYTES = 604298
ZIP_SHA256 = "139e9908f0a3544bb222386855c9ce107e96467306bb8e4ce936aab59e7baac4"
DATASET_URL = "https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand"
ZIP_URL = "https://archive.ics.uci.edu/static/public/560/seoul%2Bbike%2Bsharing%2Bdemand.zip"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CITATION = (
    "Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5F62R."
)
HEADERS = (
    "Date",
    "Rented Bike Count",
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Seasons",
    "Holiday",
    "Functioning Day",
)
UNITS = dict(zip(HEADERS[:11], (
    "dd/MM/yyyy", "bikes", "hour (0-23)", "°C", "%", "m/s", "10m",
    "°C", "MJ/m2", "mm", "cm",
)))
LABELS = {
    "Seasons": ("Autumn", "Spring", "Summer", "Winter"),
    "Holiday": ("Holiday", "No Holiday"),
    "Functioning Day": ("No", "Yes"),
}
NOTICE = (
    CITATION + "\n\n"
    "Source: " + DATASET_URL + "\n"
    "Original archive: " + ZIP_URL + "\n"
    "License: Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
    + LICENSE_URL + "\n\n"
    "Modification: Encoding only, from Windows-1252 (CP1252) to UTF-8.\n"
    "CRLF line endings, headers, rows, units and values are unchanged.\n"
    "No imputation or aggregation was performed.\n\n"
    "These are observed rentals and historical observed weather in Asia/Seoul.\n"
    "The 295 Functioning Day=No rows record zero rentals during shutdown;\n"
    "they do not establish zero latent demand. Weather is not a future forecast.\n"
    "No forecast-quality claim is made.\n"
)


def require_identity(data, size, digest, label):
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError(f"{label} exceeds the 2 MiB limit")
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{label} does not match the pinned size and SHA-256")


def read_bounded(stream, size):
    data = bytearray()
    while len(data) <= size:
        chunk = stream.read(min(65536, size + 1 - len(data)))
        if not chunk:
            return bytes(data)
        data.extend(chunk)
    raise ValueError("Input exceeds the pinned byte count")


def read_source(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Input must be a regular CSV file")
        if info.st_size > MAX_INPUT_BYTES:
            raise ValueError("Source CSV exceeds the 2 MiB limit")
        if info.st_size != SOURCE_BYTES:
            raise ValueError("Source CSV does not match the pinned byte count")
        return read_bounded(stream, SOURCE_BYTES)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise ValueError("The pinned download must not redirect")


def download_source():
    request = urllib.request.Request(ZIP_URL, headers={"Accept-Encoding": "identity"})
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(request, timeout=30) as response:
        if response.status != 200 or response.geturl() != ZIP_URL:
            raise ValueError("Unexpected response from the pinned ZIP URL")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("Encoded HTTP responses are not accepted")
        length = response.headers.get("Content-Length")
        if length is not None and (
            re.fullmatch(r"[0-9]+", length) is None or int(length) != ZIP_BYTES
        ):
            raise ValueError("Download does not match the pinned ZIP byte count")
        data = read_bounded(response, ZIP_BYTES)
    return source_from_zip(data)


def source_from_zip(data):
    require_identity(data, ZIP_BYTES, ZIP_SHA256, "Source ZIP")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != CSV_NAME:
            raise ValueError("ZIP must contain only the exact pinned CSV member")
        member = members[0]
        if member.file_size != SOURCE_BYTES or member.flag_bits & 1:
            raise ValueError("Unexpected CSV member size or encryption")
        with archive.open(member) as stream:
            source = read_bounded(stream, SOURCE_BYTES)
    require_identity(source, SOURCE_BYTES, SOURCE_SHA256, "Source CSV")
    return source


def parse_csv(text):
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    headers = next(reader, None)
    if headers != list(HEADERS):
        raise ValueError("CSV headers do not match the original headers")
    first = None
    previous = None
    count = 0
    non_functioning = 0
    for line, row in enumerate(reader, start=2):
        if len(row) != len(HEADERS):
            raise ValueError(f"Row {line} has an unexpected field count")
        if re.fullmatch(r"[0-9]{2}/[0-9]{2}/[0-9]{4}", row[0]) is None:
            raise ValueError(f"Row {line} requires a dd/MM/yyyy date")
        if any(re.fullmatch(r"[0-9]+", row[index]) is None for index in (1, 2)):
            raise ValueError(f"Row {line} requires nonnegative integer rentals and hour")
        rentals, hour = int(row[1]), int(row[2])
        if hour > 23:
            raise ValueError(f"Row {line} hour is outside 0-23")
        current = datetime.strptime(row[0], "%d/%m/%Y").replace(hour=hour)
        if previous is not None and current != previous + timedelta(hours=1):
            raise ValueError(f"Row {line} is not the next hour; duplicates and gaps are rejected")
        for value in row[3:11]:
            if not math.isfinite(float(value)):
                raise ValueError(f"Row {line} has non-finite weather")
        for index in (11, 12, 13):
            if row[index] not in LABELS[headers[index]]:
                raise ValueError(f"Row {line} has an unknown categorical label")
        if row[13] == "No":
            if rentals != 0:
                raise ValueError(f"Row {line} has rentals on a non-functioning day")
            non_functioning += 1
        if first is None:
            first = current
        previous = current
        count += 1
    if (
        count != 8760
        or first != datetime(2017, 12, 1)
        or previous != datetime(2018, 11, 30, 23)
        or non_functioning != 295
    ):
        raise ValueError("CSV does not match the pinned period and row statistics")
    return {
        "rows": count,
        "first_local": first.isoformat(),
        "last_local": previous.isoformat(),
        "non_functioning_rows": non_functioning,
        "original_headers": headers,
    }


def prepare(source):
    require_identity(source, SOURCE_BYTES, SOURCE_SHA256, "Source CSV")
    text = source.decode("cp1252")
    normalized = text.encode("utf-8")
    require_identity(normalized, NORMALIZED_BYTES, NORMALIZED_SHA256, "Normalized CSV")
    manifest = {
        **parse_csv(text),
        "format": "chronos-seoul-bike-source/1",
        "time_zone": "Asia/Seoul",
        "source_url": DATASET_URL,
        "download_url": ZIP_URL,
        "dataset_doi": "10.24432/C5F62R",
        "citation": CITATION,
        "license": "CC-BY-4.0",
        "license_url": LICENSE_URL,
        "source_bytes": len(source),
        "source_sha256": SOURCE_SHA256,
        "source_encoding": "Windows-1252",
        "normalized_bytes": len(normalized),
        "normalized_sha256": NORMALIZED_SHA256,
        "normalized_encoding": "UTF-8",
        "normalization": "original_bytes.decode('cp1252').encode('utf-8')",
        "line_endings": "CRLF",
        "zip_bytes": ZIP_BYTES,
        "zip_sha256": ZIP_SHA256,
        "zip_member": CSV_NAME,
        "units": UNITS,
        "allowed_labels": LABELS,
    }
    return {
        CSV_NAME: normalized,
        "manifest.json": (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
        "NOTICE.txt": NOTICE.encode("utf-8"),
    }


def same_file(left, right):
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def publish(output, files):
    output.mkdir()
    directory_info = output.lstat()
    directory = None
    created = {}
    try:
        directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if not same_file(directory_info, os.fstat(directory)):
            raise ValueError("Output directory changed before writing")
        for name, data in files.items():
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=directory)
            try:
                created[name] = os.fstat(descriptor)
                stream = os.fdopen(descriptor, "wb")
            except BaseException:
                os.close(descriptor)
                raise
            with stream as writer:
                writer.write(data)
    except BaseException:
        for name, identity in created.items():
            try:
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if same_file(identity, current):
                    os.unlink(name, dir_fd=directory)
            except OSError:
                pass
        try:
            if same_file(directory_info, output.lstat()):
                output.rmdir()
        except OSError:
            pass
        raise
    finally:
        if directory is not None:
            os.close(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce the pinned UCI Seoul Bike source resources.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Original CP1252 CSV; never downloads")
    source.add_argument("--download", action="store_true", help="Explicitly download the pinned UCI ZIP")
    parser.add_argument("--output", type=Path, required=True, help="New folder in an existing parent directory")
    arguments = parser.parse_args(argv)
    try:
        if os.path.lexists(arguments.output):
            raise ValueError("Output path already exists; nothing was changed")
        data = download_source() if arguments.download else read_source(arguments.input)
        files = prepare(data)
        publish(arguments.output, files)
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile, HTTPException) as error:
        print(f"bike_source: {error}", file=sys.stderr)
        return 1
    print("Prepared the pinned Seoul Bike CSV, manifest and notice (8760 hourly rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
