import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import bike_source


ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "CoreAISamplesKit/Sources/ForecastRuntime/Resources/SeoulBike"
CSV_NAME = "SeoulBikeData.csv"
SOURCE_HASH = "373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60"
NORMALIZED_HASH = "d2f7a0f0ba95e82678ead794429a18a7f696aa3b82c9b9ec34ec47806ff0767b"
ARCHIVE_HASH = "139e9908f0a3544bb222386855c9ce107e96467306bb8e4ce936aab59e7baac4"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/560/seoul%2Bbike%2Bsharing%2Bdemand.zip"


class Response(io.BytesIO):
    def __init__(self, data, headers=None, status=200, url=ARCHIVE_URL):
        super().__init__(data)
        self.headers = {} if headers is None else headers
        self.status = status
        self.url = url
        self.read_bytes = 0

    def geturl(self):
        return self.url

    def read(self, size=-1):
        if size < 0:
            raise AssertionError("Unbounded network read")
        result = super().read(min(size, 8192))
        self.read_bytes += len(result)
        return result


class BikeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundled = {path.name: path.read_bytes() for path in RESOURCES.iterdir()}
        cls.original = cls.bundled[CSV_NAME].decode("utf-8").encode("cp1252")
        if hashlib.sha256(cls.original).hexdigest() != SOURCE_HASH:
            raise AssertionError("The bundled data cannot reproduce the pinned original CSV")
        cls.text = cls.original.decode("cp1252")
        cls.rows = list(csv.reader(io.StringIO(cls.text, newline="")))
        buffer = io.BytesIO()
        member = zipfile.ZipInfo(CSV_NAME, date_time=(2023, 5, 22, 15, 23, 0))
        member.create_system = 3
        member.external_attr = 2172649472
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(member, cls.original)
        cls.archive = buffer.getvalue()
        if hashlib.sha256(cls.archive).hexdigest() != ARCHIVE_HASH:
            raise AssertionError("The offline archive fixture does not match the original ZIP")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="bike-source-")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.source = self.folder / "original.csv"
        self.source.write_bytes(self.original)
        for target in (
            "urllib.request.OpenerDirector.open",
            "socket.create_connection",
            "socket.getaddrinfo",
        ):
            blocker = mock.patch(target, side_effect=AssertionError("Real network access is forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def run_cli(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return bike_source.main(list(map(str, arguments)))

    def changed_csv(self, row, column, value):
        rows = [list(values) for values in self.rows]
        rows[row][column] = value
        return self.csv_text(rows)

    def csv_text(self, rows):
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\r\n").writerows(rows)
        return buffer.getvalue()

    def test_exact_reproduction_and_source_immutability(self):
        self.assertEqual(set(self.bundled), {CSV_NAME, "manifest.json", "NOTICE.txt"})
        self.assertEqual(len(self.original), 604166)
        self.assertEqual(len(self.bundled[CSV_NAME]), 604168)
        self.assertEqual(hashlib.sha256(self.bundled[CSV_NAME]).hexdigest(), NORMALIZED_HASH)
        self.assertEqual(self.bundled[CSV_NAME].count(b"\r\n"), 8761)
        self.assertNotIn(b"\n", self.bundled[CSV_NAME].replace(b"\r\n", b""))
        self.assertEqual(self.text.count("°"), 2)
        with mock.patch.object(bike_source, "download_source", side_effect=AssertionError("Unexpected download")):
            for name in ("first", "second"):
                output = self.folder / name
                self.assertEqual(self.run_cli("--input", self.source, "--output", output), 0)
                self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, self.bundled)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_manifest_contains_parsed_statistics_identities_units_and_citation(self):
        manifest = json.loads(self.bundled["manifest.json"])
        expected = {
            "format": "chronos-seoul-bike-source/1",
            "rows": 8760,
            "time_zone": "Asia/Seoul",
            "first_local": "2017-12-01T00:00:00",
            "last_local": "2018-11-30T23:00:00",
            "non_functioning_rows": 295,
            "source_bytes": 604166,
            "source_sha256": SOURCE_HASH,
            "normalized_bytes": 604168,
            "normalized_sha256": NORMALIZED_HASH,
            "zip_bytes": 604298,
            "zip_sha256": ARCHIVE_HASH,
            "download_url": ARCHIVE_URL,
            "zip_member": CSV_NAME,
            "license": "CC-BY-4.0",
            "dataset_doi": "10.24432/C5F62R",
        }
        for key, value in expected.items():
            self.assertEqual(manifest[key], value, key)
        self.assertEqual(manifest["original_headers"], self.rows[0])
        self.assertEqual(list(self.rows[1][:3]), ["01/12/2017", "254", "0"])
        self.assertEqual(list(self.rows[-1][:3]), ["30/11/2018", "584", "23"])
        self.assertEqual(
            [manifest["units"][name] for name in self.rows[0][3:11]],
            ["°C", "%", "m/s", "10m", "°C", "MJ/m2", "mm", "cm"],
        )
        self.assertEqual(manifest["rows"], len(self.rows) - 1)
        shutdowns = [row for row in self.rows[1:] if row[13] == "No"]
        self.assertEqual(manifest["non_functioning_rows"], len(shutdowns))
        self.assertEqual({row[1] for row in shutdowns}, {"0"})
        notice = self.bundled["NOTICE.txt"].decode("utf-8")
        self.assertIn(
            "Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5F62R.", notice,
        )
        for required in ("https://creativecommons.org/licenses/by/4.0/", "CP1252", "UTF-8", "CRLF", "latent demand", "not a future forecast"):
            self.assertIn(required, notice)

    def test_unknown_and_malformed_bytes_are_rejected_before_parsing_or_publishing(self):
        mutations = (
            b"", b"not a CSV", b"\xff" * 604166,
            self.original.replace(b"254", b"255", 1), self.bundled[CSV_NAME],
        )
        with mock.patch.object(bike_source, "parse_csv") as parse, mock.patch.object(bike_source, "publish") as publish:
            for index, data in enumerate(mutations):
                with self.subTest(index=index):
                    self.source.write_bytes(data)
                    output = self.folder / str(index)
                    self.assertEqual(self.run_cli("--input", self.source, "--output", output), 1)
                    self.assertFalse(output.exists())
            parse.assert_not_called()
            publish.assert_not_called()

    def test_oversized_file_is_rejected_before_reading(self):
        self.source.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
        with mock.patch.object(bike_source, "read_bounded") as read:
            output = self.folder / "oversized"
            self.assertEqual(self.run_cli("--input", self.source, "--output", output), 1)
            read.assert_not_called()
            self.assertFalse(output.exists())

    def test_missing_directory_and_fifo_inputs_do_not_create_output(self):
        fifo = self.folder / "fifo"
        os.mkfifo(fifo)
        for source in (self.folder / "missing", self.folder, fifo):
            with self.subTest(source=source.name):
                output = self.folder / "output"
                self.assertEqual(self.run_cli("--input", source, "--output", output), 1)
                self.assertFalse(output.exists())

    def test_short_reads_are_collected_and_overflow_is_bounded(self):
        response = Response(self.original)
        self.assertEqual(bike_source.read_bounded(response, 604166), self.original)
        response = Response(b"x" * (2 * 1024 * 1024 + 1))
        with self.assertRaises(ValueError):
            bike_source.read_bounded(response, 604298)
        self.assertEqual(response.read_bytes, 604299)

    def test_parser_rejects_changed_headers_and_row_width(self):
        for text in (
            self.text.replace("Temperature(°C)", "Temperature(C)", 1),
            self.csv_text([self.rows[0], self.rows[1][:-1], *self.rows[2:]]),
            self.csv_text([self.rows[0], [*self.rows[1], "extra"], *self.rows[2:]]),
            self.text + "\r\n",
        ):
            with self.assertRaises(ValueError):
                bike_source.parse_csv(text)

    def test_parser_requires_strict_dates_integer_counts_and_hours(self):
        for column, values in (
            (0, ("1/12/2017", "01/2/2017", "31/02/2017", "2017-12-01", "01/12/2017x")),
            (1, ("-1", "1.5", "NaN", "", "+1")),
            (2, ("24", "-1", "0.0", "", " 0")),
        ):
            for value in values:
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    bike_source.parse_csv(self.changed_csv(1, column, value))

    def test_parser_rejects_duplicates_gaps_unordered_and_incomplete_periods(self):
        for rows in (
            [self.rows[0], self.rows[1], self.rows[1], *self.rows[3:]],
            [self.rows[0], self.rows[1], *self.rows[3:]],
            [self.rows[0], self.rows[2], self.rows[1], *self.rows[3:]],
            self.rows[:-1],
            [self.rows[0], *self.rows[2:]],
        ):
            with self.assertRaises(ValueError):
                bike_source.parse_csv(self.csv_text(rows))

    def test_parser_requires_finite_weather_in_every_weather_column(self):
        for column in range(3, 11):
            for value in ("NaN", "inf", "-inf", "1e999", "missing", ""):
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    bike_source.parse_csv(self.changed_csv(1, column, value))

    def test_parser_rejects_unknown_labels_and_inconsistent_shutdowns(self):
        for column in (11, 12, 13):
            with self.subTest(column=column), self.assertRaises(ValueError):
                bike_source.parse_csv(self.changed_csv(1, column, "unknown"))
        shutdown = next(index for index, row in enumerate(self.rows[1:], 1) if row[13] == "No")
        with self.assertRaisesRegex(ValueError, "rentals on a non-functioning day"):
            bike_source.parse_csv(self.changed_csv(shutdown, 1, "1"))
        with self.assertRaisesRegex(ValueError, "row statistics"):
            bike_source.parse_csv(self.changed_csv(shutdown, 13, "Yes"))

    def test_exact_archive_and_csv_identities_are_both_checked(self):
        self.assertEqual(len(self.archive), 604298)
        self.assertEqual(bike_source.source_from_zip(self.archive), self.original)
        with mock.patch.object(bike_source, "SOURCE_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Source CSV"):
                bike_source.source_from_zip(self.archive)

    def test_unknown_malformed_oversized_and_traversal_archives_are_not_opened(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("../SeoulBikeData.csv", b"x" * (2 * 1024 * 1024 + 1))
        for data in (b"not a ZIP", self.archive[:-1], self.archive + b"x", buffer.getvalue(), b"x" * (2 * 1024 * 1024 + 1)):
            with mock.patch.object(bike_source.zipfile, "ZipFile") as open_zip:
                with self.assertRaises(ValueError):
                    bike_source.source_from_zip(data)
                open_zip.assert_not_called()

    def test_explicit_download_reproduces_the_same_bundle_with_mock_http(self):
        for index, headers in enumerate(({}, {"Content-Length": "604298", "Content-Encoding": "identity"})):
            response = Response(self.archive, headers=headers)
            opener = mock.Mock()
            opener.open.return_value = response
            with mock.patch.object(bike_source.urllib.request, "build_opener", return_value=opener):
                output = self.folder / str(index)
                self.assertEqual(self.run_cli("--download", "--output", output), 0)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, self.bundled)
            request = opener.open.call_args.args[0]
            self.assertEqual(request.full_url, ARCHIVE_URL)
            self.assertEqual(request.get_header("Accept-encoding"), "identity")
            self.assertEqual(opener.open.call_args.kwargs, {"timeout": 30})
            self.assertTrue(response.closed)

    def test_download_rejects_unexpected_headers_status_and_location_before_body(self):
        responses = [
            Response(self.archive, headers={"Content-Length": value})
            for value in ("2097153", "604297", "-1", "unknown")
        ]
        responses.extend((
            Response(self.archive, headers={"Content-Encoding": "gzip"}),
            Response(self.archive, status=206),
            Response(self.archive, url="https://example.invalid/mirror.zip"),
        ))
        for index, response in enumerate(responses):
            opener = mock.Mock()
            opener.open.return_value = response
            with mock.patch.object(bike_source.urllib.request, "build_opener", return_value=opener):
                output = self.folder / str(index)
                self.assertEqual(self.run_cli("--download", "--output", output), 1)
            self.assertFalse(output.exists())
            self.assertEqual(response.read_bytes, 0)
            self.assertTrue(response.closed)

    def test_download_rejects_truncated_unknown_and_oversized_bodies(self):
        for index, data in enumerate((self.archive[:-1], b"x" * 604298, b"x" * (2 * 1024 * 1024 + 1))):
            response = Response(data)
            opener = mock.Mock()
            opener.open.return_value = response
            with mock.patch.object(bike_source.urllib.request, "build_opener", return_value=opener):
                output = self.folder / str(index)
                self.assertEqual(self.run_cli("--download", "--output", output), 1)
            self.assertLessEqual(response.read_bytes, 604299)
            self.assertFalse(output.exists())
            self.assertTrue(response.closed)

    def test_redirects_are_refused(self):
        with self.assertRaisesRegex(ValueError, "must not redirect"):
            bike_source.NoRedirect().redirect_request(None, None, 302, "Found", {}, ARCHIVE_URL)

    def test_input_selection_is_required_and_mutually_exclusive(self):
        for flags in ((), ("--input", self.source, "--download")):
            with self.assertRaises(SystemExit) as result:
                self.run_cli(*flags, "--output", self.folder / "output")
            self.assertEqual(result.exception.code, 2)

    def test_existing_destinations_including_symlinks_are_preserved_without_download(self):
        empty = self.folder / "empty"
        empty.mkdir()
        full = self.folder / "full"
        full.mkdir()
        (full / "keep.txt").write_bytes(b"keep")
        link = self.folder / "link"
        link.symlink_to(self.source)
        broken = self.folder / "broken"
        broken.symlink_to(self.folder / "absent")
        with mock.patch.object(bike_source, "download_source", side_effect=AssertionError("Unexpected download")):
            for output in (empty, full, self.source, link, broken):
                self.assertEqual(self.run_cli("--download", "--output", output), 1)
        self.assertEqual(list(empty.iterdir()), [])
        self.assertEqual((full / "keep.txt").read_bytes(), b"keep")
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(os.readlink(link), str(self.source))
        self.assertTrue(broken.is_symlink())

    def test_destination_created_after_initial_check_is_preserved(self):
        output = self.folder / "raced"
        prepared = bike_source.prepare(self.original)

        def create_destination(data):
            output.mkdir()
            (output / "keep.txt").write_bytes(b"keep")
            return prepared

        with mock.patch.object(bike_source, "prepare", side_effect=create_destination):
            self.assertEqual(self.run_cli("--input", self.source, "--output", output), 1)
        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, {"keep.txt": b"keep"})

    def test_output_parent_follows_filesystem_traversal_through_alias(self):
        parent = self.folder / "real"
        nested = parent / "nested"
        nested.mkdir(parents=True)
        alias = self.folder / "alias"
        alias.symlink_to(nested, target_is_directory=True)
        trap = self.folder / "prepared"
        trap.write_bytes(b"keep")
        output = alias / ".." / "prepared"
        self.assertEqual(self.run_cli("--input", self.source, "--output", output), 0)
        self.assertEqual({path.name: path.read_bytes() for path in (parent / "prepared").iterdir()}, self.bundled)
        self.assertEqual(trap.read_bytes(), b"keep")

    def test_write_failure_removes_only_this_attempts_output(self):
        output = self.folder / "failed"
        real_open = os.open

        def fail_manifest(path, *args, **kwargs):
            if path == "manifest.json":
                raise OSError("Injected write failure")
            return real_open(path, *args, **kwargs)

        with mock.patch.object(bike_source.os, "open", side_effect=fail_manifest):
            self.assertEqual(self.run_cli("--input", self.source, "--output", output), 1)
        self.assertFalse(output.exists())
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_partial_file_is_cleaned_up_after_write_failure(self):
        output = self.folder / "partial"
        real_fdopen = os.fdopen

        @contextlib.contextmanager
        def failing_stream(descriptor, mode):
            with real_fdopen(descriptor, mode) as stream:
                stream.write(b"partial")
                writer = mock.Mock()
                writer.fileno.return_value = stream.fileno()
                writer.write.side_effect = OSError("Injected partial write failure")
                yield writer

        with mock.patch.object(bike_source.os, "fdopen", side_effect=failing_stream):
            with self.assertRaises(OSError):
                bike_source.publish(output, self.bundled)
        self.assertFalse(output.exists())

    def test_stream_open_failure_closes_and_removes_the_created_file(self):
        output = self.folder / "stream-failed"
        descriptors = []

        def fail_stream(descriptor, mode):
            descriptors.append(descriptor)
            raise OSError("Injected stream setup failure")

        with mock.patch.object(bike_source.os, "fdopen", side_effect=fail_stream):
            with self.assertRaises(OSError):
                bike_source.publish(output, self.bundled)
        self.assertFalse(output.exists())
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])

    def test_failure_preserves_unrelated_files_and_replaced_output_entries(self):
        for replace in (False, True):
            output = self.folder / str(replace)
            real_open = os.open

            def fail_manifest(path, *args, **kwargs):
                if path == "manifest.json":
                    (output / "keep.txt").write_bytes(b"unrelated")
                    if replace:
                        (output / CSV_NAME).rename(self.folder / "moved.csv")
                        (output / CSV_NAME).symlink_to(self.source)
                    raise OSError("Injected write failure")
                return real_open(path, *args, **kwargs)

            with mock.patch.object(bike_source.os, "open", side_effect=fail_manifest):
                self.assertEqual(self.run_cli("--input", self.source, "--output", output), 1)
            self.assertEqual((output / "keep.txt").read_bytes(), b"unrelated")
            if replace:
                self.assertTrue((output / CSV_NAME).is_symlink())
            else:
                self.assertEqual({path.name for path in output.iterdir()}, {"keep.txt"})
        self.assertEqual(self.source.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
