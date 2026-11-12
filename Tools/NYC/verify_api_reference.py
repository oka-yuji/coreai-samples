import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq


def verify(raw, reference):
    data = reference.read_bytes()
    if hashlib.sha256(data).hexdigest() != '2da47735c51b5987dec9f6bee72c97acd39abe4870991e413dca30a1912c6530':
        raise ValueError('Reference hash mismatch')
    value = json.loads(data)
    if value['format'] != 'nyc-verified-counts/1' or value['timeZone'] != 'America/New_York':
        raise ValueError('Unsupported reference contract')
    if value['start'] != '2025-01-01T00:00:00-05:00' or value['end'] != '2025-02-01T00:00:00-05:00':
        raise ValueError('This verifier supports the pinned January reference')
    if value['intervalSeconds'] != 1800 or len(value['series']) != 20:
        raise ValueError('Unexpected reference shape')
    areas = [row['areaID'] for row in value['series']]
    if len(set(areas)) != len(areas) or any(len(row['values']) != 1488 for row in value['series']):
        raise ValueError('Invalid reference inventory')
    start = np.datetime64(datetime(2025, 1, 1), 'us')
    end = np.datetime64(datetime(2025, 2, 1), 'us')
    counts = {area: np.zeros(1488, dtype=np.int64) for area in areas}
    sources = []
    for source in value['sourceParquets']:
        url = urlsplit(source['url'])
        name = Path(url.path).name
        if url.scheme != 'https' or url.netloc != 'd37ci6vzurychx.cloudfront.net':
            raise ValueError('Unexpected source host')
        if name not in ['yellow_tripdata_2025-01.parquet', 'yellow_tripdata_2025-02.parquet']:
            raise ValueError('Unexpected source file')
        path = raw / name
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != source['sha256']:
            raise ValueError('Source hash mismatch: ' + name)
        table = pq.read_table(path, columns=['tpep_pickup_datetime', 'PULocationID'])
        times = table['tpep_pickup_datetime'].to_numpy().astype('datetime64[us]')
        zones = table['PULocationID'].to_numpy()
        included = (times >= start) & (times < end)
        for area in areas:
            selected = times[included & (zones == int(area))]
            indices = ((selected - start).astype('timedelta64[us]').astype(np.int64) // 1_800_000_000)
            counts[area] += np.bincount(indices, minlength=1488)
        sources.append({'file': name, 'sha256': digest, 'rows': len(times)})
    if {source['file'] for source in sources} != {
        'yellow_tripdata_2025-01.parquet', 'yellow_tripdata_2025-02.parquet'
    } or len(sources) != 2:
        raise ValueError('Both pinned source files are required')
    for row in value['series']:
        expected = np.asarray(row['values'], dtype=np.int64)
        if not np.array_equal(expected, counts[row['areaID']]):
            raise ValueError('Count mismatch for area ' + row['areaID'])
    return {'passed': True, 'values': 29760, 'reference_sha256': hashlib.sha256(data).hexdigest(),
            'sources': sources, 'scope': 'January pickup intervals across both monthly source files; no live API claim'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[2] /
                        'CoreAISamplesKit/Sources/ChronosDemoRuntime/Resources/NYCVerified/january-2025.json')
    args = parser.parse_args()
    print(json.dumps(verify(args.raw, args.reference), indent=2))


if __name__ == '__main__':
    main()
