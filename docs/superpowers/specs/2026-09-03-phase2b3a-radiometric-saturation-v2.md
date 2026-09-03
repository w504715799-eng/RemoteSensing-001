# Phase 2B3-A Radiometric Saturation v2 Design

## Context

The exact 120-ROI development run stopped before model construction on sample
`NA5120_E1241N0607__m_3311808_ne_11_060_20220512`. Its frozen LR and HR GeoTIFFs and sidecars
agree on a maximum raw value of `11968`. The aligned crops contain eight LR and 117 HR values above
`10000`, in bands B04 and B08. The other 119 development ROI have maxima no greater than `9572`.

The existing `uint16_divide_10000_no_clip_v1` policy was based on a 12-pair smoke whose observed
range was `[20,9572]`. Its design explicitly required a policy review if real data exceeded `10000`.
The frozen dataset metadata does not retain a Sentinel-2 processing baseline, quantification value,
additive offset, or harmonization marker, so inventing an offset is not supportable. The assets pass
all byte, path, GeoTIFF, shape, dtype, CRS, nodata, mask, and sidecar-integrity checks.

## Decision

Phase 2B3-A will use a versioned, explicit saturation policy:

`uint16_saturate_10000_divide_10000_v2`

For each aligned LR and HR crop:

1. Preserve all existing byte, sidecar, geometry, dtype, nodata, and mask checks.
2. Reject raw values above `32767`, the documented upper DN bound for post-PB04 Sentinel-2 image
   samples. Values equal to the frozen nodata sentinel `65535` remain invalid through the mask and
   nodata checks.
3. Count values strictly greater than `10000`, both in total and independently for the ordered bands
   B04, B03, B02, and B08.
4. Saturate only the aligned crop with `min(raw, 10000)`.
5. Convert to contiguous CPU `torch.float32` and divide by `10000.0`.

This keeps model inputs in their existing `[0,1]` domain, retains the exact frozen 120-ROI sample,
and makes every changed value visible in host-free evidence. No sample is removed or replaced, no
unknown offset is applied, and no calibration or `internal_test` pixels are inspected.

## Versioning and historical evidence

The legacy `uint16_divide_10000_no_clip_v1` loader remains available as the default for historical
Phase 2B2-A/2B2-B code paths. The Phase 2B3-A CLI explicitly requests v2. Existing tracked A1
publication JSON and the Phase 2B2-A input audit remain byte-for-byte unchanged and continue to be
historical evidence for their producer commits.

Phase 2B3-A A1 is rerun under v2 from a clean live `phase2b3a` directory. Its result, cache audit,
replay, and runtime use v2 schemas and record zero or nonzero saturation statistics. A2 accepts only
the new v2 A1 evidence. Prediction cache provenance includes the v2 policy string, so old v1 cache
entries cannot satisfy v2 identities even when a tensor happens to be byte-identical.

The restored immutable A1 checkpoint remains untouched. After restore, its disposable live
`trustsr/phase2b3a` directory may be deleted with exact guarded paths and recreated empty; the
durable checkpoint is the recovery source. This avoids collisions and prevents v1 result or cache
reuse before the v2 A1 rerun.

## Data structures

`RadiometricSaturation` is an immutable record attached independently to LR and HR metadata:

```python
@dataclass(frozen=True)
class RadiometricSaturation:
    raw_crop_minimum: int
    raw_crop_maximum: int
    clipped_high_count: int
    clipped_high_by_band: tuple[int, int, int, int]
```

Every Phase 2B3-A A1/A2 sample record includes:

```json
{
  "radiometric_saturation": {
    "lr": {
      "raw_crop_minimum": 208,
      "raw_crop_maximum": 11968,
      "clipped_high_count": 8,
      "clipped_high_by_band": [4, 0, 0, 4]
    },
    "hr": {
      "raw_crop_minimum": 208,
      "raw_crop_maximum": 11968,
      "clipped_high_count": 117,
      "clipped_high_by_band": [56, 0, 0, 61]
    }
  }
}
```

Each result also contains an aggregate `radiometric_policy` object with the exact policy name, raw
upper bound `32767`, saturation threshold `10000`, ordered band names, affected sample/asset counts,
LR and HR clipped-value totals, and the maximum observed crop value. Aggregates are derived from the
sample records and independently revalidated during replay and offline bundle verification.

## Provenance and schemas

- New A1 result, cache-audit, runtime, replay, and bundle schemas use version `v2`.
- A2 schemas remain pre-publication `v1`, but require `normalization_policy` and
  `radiometric_policy`; this is not a migration of published A2 evidence because no A2 evidence
  exists.
- A2 runtime records the current v2 A1 producer commit and replay digest through the existing
  ancestry-bound fields.
- The old Phase 2B2-A input-audit digest remains an upstream structural and repeatability input. It
  is not relabelled as v2 and does not substitute for the new per-sample radiometric evidence.
- Offline verification recomputes all saturation aggregates from the exact 120 result samples and
  rejects missing, inconsistent, non-integer, negative, wrong-band-length, wrong-policy, or
  out-of-domain values.

## Failure behavior

The loader fails before model construction if an asset or crop violates integrity, contains invalid
mask/nodata data, exceeds raw DN `32767`, requests an unknown normalization policy, or produces
inconsistent saturation counts. A1/A2 replay and offline verification fail closed on any policy or
aggregate mismatch. Existing atomic result/cache publication semantics are unchanged.

## Testing

- TDD regression with real temporary GeoTIFFs containing crop values `10001`, `11288`, and `11968`.
- Legacy v1 still rejects the same fixture.
- v2 saturates exact positions, leaves source arrays/files unchanged, reports exact per-band counts,
  and rejects `32768`.
- Phase 2B3-A pair loading explicitly requests v2; Phase 2B2-A/2B2-B defaults remain v1.
- Prediction identities differ between v1 and v2.
- A1/A2 result, replay, runtime, bundle, and offline-verifier tests cover exact radiometric objects
  and tampering failures.
- Full pytest, Ruff, shell syntax, CLI help, and `git diff --check` run before cloud use.

## Cloud rerun

After the reviewed repair is merged and pushed, restore the immutable A1 checkpoint only to recover
the frozen data/model workspace. Guardedly remove the disposable restored `trustsr/phase2b3a`,
recreate it empty, run formal preflight, then run v2 `single`, `smoke`, `replay`, and checkpoint a new
A1. Only after new A1 acceptance succeeds may `development`, `development-replay`, and the A2
checkpoint run. The GPU remains off during local implementation and review.
