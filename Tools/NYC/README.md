# Fixed NYC pickup records

This demo uses NYC Taxi & Limousine Commission Yellow Taxi records for January 2025 and simplified NYC taxi-zone geometry. Counts describe recorded pickups, not passengers, latent demand or dispatch recommendations. Vendors supplied the records; NYC TLC does not guarantee their accuracy. This app is independent of NYC and does not imply endorsement.

The [provenance record](provenance.json) identifies the reviewed public-data policy, statute and City terms. Do not label the data Apache-2.0 or CC0. Model licensing is separate. Only derived aggregates, simplified geometry and precomputed predictions are included; raw trips, GIS archives and model weights stay outside Git.

`spec.json` fixes the source URLs, byte sizes, SHA-256 hashes, original acquisition timestamps, 20 Manhattan zone IDs, three origins, America/New_York timezone and aggregation rules. Each pickup belongs to the half-open local-time interval `[start, start + 30 minutes)`. All days here are in January, with UTC−05:00 and no daylight-saving transition. Every row counts once, independent of passenger count or fare. Missing bins become zero observed pickups. Invalid/out-of-month timestamps and other zones are excluded. The model receives only one zone's count sequence; geometry and zone IDs never enter the model.

Each scene contains 1,008 history bins and 16 future bins. Geometry starts in EPSG:2263, is simplified by 15 feet preserving topology, transformed to EPSG:4326 and rounded to six decimal places. Holes and a representative point are retained.

Use Python 3.12.12 and the exact packages in `requirements.txt` / `environment.json`. Preparation and verification are offline; supply the pinned raw files separately:

```sh
python Tools/NYC/prepare.py --raw /path/to/private/raw --output /new/derived
python Tools/NYC/verify.py --raw /path/to/private/raw --data CoreAISamplesKit/Sources/ChronosDemoRuntime/Resources/NYCData
python Tools/NYC/verify.py --data CoreAISamplesKit/Sources/ChronosDemoRuntime/Resources/NYCData
```

The first command recreates the original eight JSON files, whose hashes are in `resources.json`. It intentionally preserves the original snapshot's acquisition timestamps. A fresh download must record its actual acquisition timestamp separately; running this tool is not a new acquisition. Compare the eight generated hashes before adding the pinned `forecasts.json` to a reproduction folder. The second command independently groups all pickup timestamps with Arrow, comparing all 61,440 history/truth values; preparation uses NumPy bin indexing and `add.at`. The data-only command checks all nine exact files and the cache's source/model identity; it does not claim independent aggregation verification.

`forecasts.json` has 39 frames (three scenes × origins 0…12), 20 series per frame, and quantile indices 2/10/18 projected to the first four steps. Its graph hash is `3094871fc431ae5d86d07c8b487ab53e48e60e9fb56efe803ba18c5bf88889c8`. Supplied generation evidence compared all 262,080 raw outputs from 780 requests to official FP32 references at `abs(error) <= 1e-4 + 1e-4 * abs(reference)`. Worker CPU checks validate the supplied bytes and identities; they do not repeat those model executions.

To regenerate predictions, the parent must hold the repository's shared heavy-job lock and run the existing `chronos-verify --bundle … --input … --output …` for each scene, origin and zone. Each input is `{"context": (history + truth[:origin])[-1008:]}`. Require the current manifest and graph identity in each output, compare all 21×16 values against official FP32 results at the frozen tolerance, then retain only indices 2/10/18 and the first four steps in the preview file. Record the new actual generation time separately and review any hash update. Do not relabel the supplied generation timestamp or invent cached device timings.

The native Models integration test and parent checks are documented in `docs/chronos2.md`. CPU checks do not certify model fidelity, the Apple basemap, GUI interaction or iPhone memory behavior.

## Verified API reference

The shipped API reference covers the twenty demo zones and all 1,488 half-hour pickup intervals in January 2025. It pins dataset `irx2-vgm4` at revision `2026-10-01T16:57:47.000Z` and reference SHA-256 `2da47735c51b5987dec9f6bee72c97acd39abe4870991e413dca30a1912c6530`.

The January file alone differs from the annual API at nine month-end cells. The February file contains thirty January pickup records, ten in the selected zones; including those ten resolves all 29,760 counts. Both raw sources are required for reproduction. The initial mismatch remains in the validation history. Monthly filenames are not exact pickup-time boundaries.

Place the pinned January and February Parquet files in a private directory and use the Python environment in `environment.json`:

```sh
python Tools/NYC/verify_api_reference.py --raw /path/to/raw
```

The command hashes both sources and independently reaggregates their January pickup columns. It does not contact the API or claim that an unknown revision or month is verified. Raw trips stay outside Git. The reference JSON carries both official source URLs and hashes; NYC TLC attribution and the data-use qualifications above also apply.
