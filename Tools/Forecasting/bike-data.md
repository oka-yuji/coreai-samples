# Seoul Bike source data

This prepares the independent source-data portion of Issue #530. It includes no Swift target, UI, model execution, weather forecast or forecast-quality result. Forecast integration and its acceptance checks are separate work.

## Source and attribution

Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5F62R.

- [UCI dataset page](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
- [Original UCI ZIP](https://archive.ics.uci.edu/static/public/560/seoul%2Bbike%2Bsharing%2Bdemand.zip)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The UCI source metadata lists no individual creators. The bundled `NOTICE.txt` preserves the repository citation, source and license links, and identifies the encoding-only modification. These data retain their CC BY 4.0 license independently of the repository's software license.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Original ZIP | 604298 | `139e9908f0a3544bb222386855c9ce107e96467306bb8e4ce936aab59e7baac4` |
| Original `SeoulBikeData.csv` member | 604166 | `373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60` |
| Bundled UTF-8 `SeoulBikeData.csv` | 604168 | `d2f7a0f0ba95e82678ead794429a18a7f696aa3b82c9b9ec34ec47806ff0767b` |

## Reproduce

Run from the repository root using Python 3 and its standard library. File-creation safeguards use POSIX descriptor operations; Windows support is not provided.

```sh
python3 -B Tools/Forecasting/bike_source.py --input /path/to/original.csv --output /new/folder
```

The output's parent directory must already exist. The output path itself must not exist, including an empty directory, file or dangling symlink. No input file is modified. Each successful run creates exactly `SeoulBikeData.csv`, `manifest.json` and `NOTICE.txt`, byte-identical to `CoreAISamplesKit/Sources/ForecastRuntime/Resources/SeoulBike/`.

An explicit download is also supported:

```sh
python3 -B Tools/Forecasting/bike_source.py --download --output /new/folder
```

`--input` and `--download` are required alternatives. Local input never triggers a download. The download uses only the pinned HTTPS ZIP URL, rejects redirects and encoded HTTP bodies, and checks the complete ZIP identity before opening its sole, exact CSV member. It never extracts archive paths or executes content. Both paths validate the original CSV identity before decoding or parsing. Unknown bytes fail closed; there is no hash override, alternate source or mirror option.

The hard input limit is 2 MiB. Known sizes narrow reads further: the CSV and ZIP readers consume at most their pinned size plus one byte to detect overflow. Local file sizes and supplied HTTP Content-Length are checked before reading; absent HTTP lengths still use bounded reads. A download timeout is 30 seconds per blocking operation. No network request is made by the tests.

## Encoding and data contract

The only transformation is:

```python
original_bytes.decode("cp1252").encode("utf-8")
```

The original header contains two degree signs. Re-encoding adds two bytes, preserves all 8,761 CRLF line endings, and changes no header, cell, row order or unit. No sorting, imputation, aggregation, resampling, clamping or newline normalization occurs.

The parser independently obtains the row count, first/last local timestamps and non-functioning count. It validates the original 14 headers and all 8,760 consecutive hourly records, from `2017-12-01T00:00:00` through `2018-11-30T23:00:00`, interpreted in `Asia/Seoul`. Dates must have the complete `dd/MM/yyyy` form and be valid calendar dates. Hours are integers from 0 through 23; rentals are nonnegative integers. Duplicates, gaps, reordered rows, missing fields, unknown labels and non-finite weather values are rejected.

| Original column | Unit or allowed values |
| --- | --- |
| Date | `dd/MM/yyyy`, Asia/Seoul local date |
| Rented Bike Count | bikes |
| Hour | integer hour, 0–23 |
| Temperature(°C) | °C |
| Humidity(%) | % |
| Wind speed (m/s) | m/s |
| Visibility (10m) | 10m |
| Dew point temperature(°C) | °C |
| Solar Radiation (MJ/m2) | MJ/m2 |
| Rainfall(mm) | mm |
| Snowfall (cm) | cm |
| Seasons | Autumn, Spring, Summer, Winter |
| Holiday | Holiday, No Holiday |
| Functioning Day | No, Yes |

All 295 `Functioning Day=No` rows have zero recorded rentals. This describes observed rentals during shutdown, not zero latent demand. Weather columns contain historical observations, not forecasts available for future hours. They must not be treated as a future weather forecast or used to claim forecast quality.

`manifest.json` records parsed statistics, original headers and units, allowed labels, source URLs, citation/license, and pinned CSV/normalized/ZIP byte identities. The ZIP identity describes the pinned distribution even when local CSV input is used; it is not a claim that a download occurred. UTF-8 JSON uses sorted keys, two-space indentation and a final LF. Metadata contains no runtime timestamp, local path or random value.

All validation and output serialization finish in memory before the new output directory is exclusively created. Files use exclusive creation. On a caught write failure, cleanup considers only files this attempt created, checks their filesystem identity, and removes the directory only if its identity is unchanged and it is empty. Unrelated or replaced entries are preserved. This is not an atomic whole-directory transaction: a process crash or forced termination can leave a partial new directory, which a later run will refuse to overwrite.

## Offline verification

```sh
python3 -B -I -m unittest discover -s Tools/Forecasting -p test_bike_source.py -v
```

Temporary fixtures stay in `.sf-tmp/`. Tests reverse only the encoding of the bundled CSV and verify the independent original hash. They reconstruct the original ZIP container with its fixed member metadata and verify its independent archive hash. HTTP responses are mocked and real network entry points are blocked. Tests cover exact repeated output, original immutability, metadata/units/attribution, malformed and oversized inputs, strict row semantics, ZIP rejection before parsing, bounded HTTP reads, destination preservation, and write-failure cleanup.

In the supervised workspace, the protected verifier additionally reads the original UCI CSV independently, runs the real CLI twice, compares every output byte with the bundled resources, checks overwrite/unknown/oversized refusal and source immutability, and invokes the offline tests:

```sh
python3 -I _sf/verify_bike.py . . .sf-build/check-1
```

Use a fresh verification directory for each attempt. This verifies source preparation only. Live UCI download, the Swift parser, Forecast UI, native inference, Mac/iPhone acceptance and final repository reviews remain outside these checks.
