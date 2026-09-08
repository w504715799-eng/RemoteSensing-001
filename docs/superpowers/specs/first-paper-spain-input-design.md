# Spain decoded-input contract

2026-09-08, implementation specification; not an external access permit or final freeze.

Module `src/trustsr/data/spain_inputs.py` accepts already-decoded NumPy arrays only. It does not
download, unpickle, inspect a dataset package or access models. A separate authenticated decoder
and embedded-member matcher are still required before real use.

`prepare_spain_pair(sample_id, lr_l2a, hr_harmonized)` requires exact shapes (12,128,128) and
(4,512,512), native integer/float arrays, finite selected values in [0,32767]. Select L2A indices
[3,2,1,7]; HRharm is already RGBN. Return CPU float32 SRPair and saturation counts.
No crop for primary R1/R9: use the declared 512×512 benchmark patch. OpenSR's separate 16-pixel
diagnostic crop does not change the primary risk grid.

Proposed study normalization: clip selected reflectance-DN above 10000, divide by 10000, matching
the development normalization. This is our processing rule, not a claim that Spain values or nodata
semantics have been verified. Reject negatives, nonfinite values and 65535; do not turn missing
pixels into zeros, interpolate or filter by quality. Zero is preserved as a valid numeric value,
not treated as a nodata identifier without source evidence. Unknown source nodata remains a
provenance limitation that must be resolved or disclosed before freeze.

Tests before implementation: band order with hand constants; clipping counts; shape/type/range
rejection; preservation of inputs; whole-patch shape. No HR-dependent score generation.
This module alone cannot authorize or validate pickle loading or tensor/member correspondence.

`bind_member_rows(embedded_dataframe, expected_projected_members)` matches all six identity/source/
coordinate fields, rejecting duplicate IDs or missing fields. Return tensor row positions in the
expected list's order; never use DataFrame index labels or assume rows are sorted. Extra quality
columns are ignored and not exported. Test reversed rows with duplicate index labels, plus source,
affine and membership mismatches. This verifies metadata correspondence, not source ground truth.
