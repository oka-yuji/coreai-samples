# Forecast source reproducibility

The product-sales demo uses [Hierarchical Sales Data, Mendeley Data v1](https://doi.org/10.17632/njdkntcpc9.1). The original provider specifies **CC BY-NC 3.0**; [UCI](https://archive.ics.uci.edu/dataset/611/hierarchical+sales+data) displays CC BY 4.0. Apply the original noncommercial terms. This repository does not redistribute the CSV, sales observations, derived fixtures or native reports containing those observations.

Explicit acquisition into a private, ignored location:

```sh
python3 Tools/Forecasting/sales_source.py --output .local/forecast/sales.csv
python3 Tools/Forecasting/sales_source.py --check .local/forecast/sales.csv
```

The script follows provider redirects, limits the response to 2 MiB, verifies the 888,985 bytes and SHA-256 `0dfd4a5bf801bf79ddb5fb1f6bc8d023487c36dd6a2a70b312cdfa9fa6568d83`, and refuses existing output. It never runs a model. The app has the same identity gate and acquires data only after **Load public sales data**. It revalidates its own cache on every read. A failed check is visible; retrying the load fetches a new copy. Valid cached data works without a network request.

Offline script tests use synthetic bytes only:

```sh
python3 -m unittest discover -s Tools/Forecasting -p 'test_*.py' -v
```

See [the demo contract](../../docs/chronos2-sales.md) for calendar rules, evaluation boundaries and the opt-in native test. Official reference outputs must be generated independently from the exact request contexts, the pinned official Chronos-2 FP32 revision and its full 21 × 16 output. Do not generate references from the implementation under test or commit source-derived reports.
