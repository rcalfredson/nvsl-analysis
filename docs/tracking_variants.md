# Offline tracking variants

Original acquisition tracking remains the default. Native selection opens the
original video, `.data`, and `__note.txt` through the existing acquisition paths;
only the trajectory payload changes. No staging, symlinks, copying companions,
or offline-tracking Python package is required.

```sh
python analyze.py -v /absolute/session.avi -f 0-1 \
  --tracking-variant body-core --tracking-manifest /absolute/tracking.json
```

Version 1 manifests contain a `recordings` object keyed by **full original video
paths**, with variant names mapped to existing companion `.trx` paths:

```json
{
  "schema_version": 1,
  "recordings": {
    "/recordings/2025-07-17/morning/c41__2025-07-17__12-26-36.avi": {
      "body-core": "/tracking/run-01/c41.body-core.offline.trx"
    },
    "/recordings/2025-07-17/night/c41__2025-07-17__20-00-00.avi": {
      "body-core": "companions/night.body-core.trx"
    }
  }
}
```

Video keys must be absolute. Paths are canonicalized, including symlinks; duplicate
canonical recording keys are rejected. Companion paths may be absolute or relative
to the manifest. Camera/day or basename matching is never used, so multiple sessions
on the same camera/day remain distinct. `original` is reserved for acquisition
tracking and ignores the manifest.

A named variant is strict by default: a missing recording/variant entry is an
error. `--tracking-fallback` explicitly permits original tracking for absent entries.
A manifest is still required. An entry with a missing, corrupt, or invalid companion
always errors, including in fallback mode. Use `--no-tracking-fallback` to override
a local fallback default.

Machine defaults use the existing `.analyze.local.env` reader in the working
directory (this is a file, not shell environment variable lookup):

```dotenv
TRACKING_VARIANT=body-core
TRACKING_MANIFEST=/absolute/tracking.json
TRACKING_FALLBACK=false
```

Each explicit CLI flag overrides its corresponding local default. Use
`--tracking-variant original` to override a local variant default.

Companions must be trusted acquisition-compatible Python pickle `.trx` files with
`x`, `y`, `w`, `h`, `theta`, `ts`, and
`tracking_metadata.source_recording`. Source metadata must contain `video`, `trx`,
and `data` identities, each with `path`, `size_bytes`, and `mtime_ns`; provided
`sha256` values are verified too. These are the identities already written by
nvsl-offline-tracking. All five trajectory arrays must match the original fly/frame
shape, and timestamps must match the original exactly. Source identity errors
require correcting the manifest or regenerating a companion from the correct
acquisition recording; fallback never hides them.

Every recording logs a `[tracking]` JSON record containing the original video,
requested and resolved variant, and selected trajectory path, size, modification
time, and SHA-256. The same record is available as `va.tracking_selection`.
In-memory keys include acquisition and selected file identities and variants.
Wall-contact cache filenames and manifests include trajectory selection, allowing
original and offline results to coexist, including with a shared cache directory.
Older wall-cache filenames are left in place and recomputed under the new keys.

## Flat-cohort fixture and verification

[test/fixtures/flat-ar-control-body-core.tracking.json](../test/fixtures/flat-ar-control-body-core.tracking.json)
contains all 34 completed companion paths from the read-only sibling repository's
`runs/flat-ar-control-body-core/status.json`. It is a machine-specific test manifest;
the held recording has no completed companion and is deliberately omitted.

Run unit coverage with the analysis environment:

```sh
python -m pytest -q test/test_tracking_selection.py test/test_wall_contact_cache.py test/test_video_analysis_wall_contact.py
```

The integration verifier below replays the saved cohort commands with original
recording paths and native selection. It writes only to a new output directory,
then compares both `learning_stats.csv` and `sli.csv` against the existing
symlink-based exports. Comparison ignores command metadata and canonicalizes the
SLI `closed_video` path to the original recording; all other fields must be exact.

```sh
python scripts/verify_native_tracking_exports.py \
  --reference ../nvsl-offline-tracking/runs/flat-ar-control-behavior \
  --output /tmp/nvsl-native-tracking-verification
```

Verification on 2026-11-19 passed for all 34 completed recordings. Offline exports
matched all 1,477 CSV records in `learning_stats.csv` and 68 in `sli.csv`; original
exports matched 1,453 and 67 respectively (counts include headers and blank CSV
records, excluding the command metadata record). Every retained field matched
exactly. The targeted unit suite passed 21 tests. Full run logs, commands, exports,
and `comparison.json` were saved under `/tmp/nvsl-native-tracking-verification`.
