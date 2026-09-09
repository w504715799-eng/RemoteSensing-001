# Public-source acceptance and metadata partition preview

User instruction on 2026-09-09 supersedes the earlier requirement to obtain an
actual per-member source mapping. A public paper or open publication is sufficient
for dataset provenance acceptance. Stop upstream contact and product-ID investigation.

- [x] Record SEN2NAIPv2's official public release and original SEN2NAIP paper;
  distinguish the v2 8,000-pair release from the v1 paper's smaller collection.
- [x] Update the study design and handoff. Geographic/NAIP grouping remains an
  operational separation rule; source-independent statistical guarantees remain
  conditional assumptions, not facts established by public availability.
- [x] Produce a deterministic metadata-only partition preview from the existing
  authenticated member projection and 360 historical identities. Use the audited
  primary 5 km connected components with shared NAIP IDs. Exclude entire touched
  components. Retain 7 km grouping as a previously reported sensitivity, not a
  second split chosen after evaluating results.
- [x] SHA-rank eligible groups with domain `trustmask-partition-preview-v1` and
  seed 3407. Allocate 256 scale-fit, 512 development-calibration, 845 development-
  validation, 1,500 calibration and 3,045 test groups. This exactly partitions the
  previously audited 6,158 groups. Keep full assignments locally in ignored
  artifacts; publish the digest, counts and limitations only. Preview is not an
  image-access permit or a frozen final execution protocol.
- [x] Test row-order invariance, disjoint roles and insufficient-group rejection;
  replay actual membership and old-data exclusion, run focused checks and review.

Next implementation phase: CPU synthetic integrated scale fitting/development
selection, group-risk calibration and paired evaluation. Freeze the resulting
method list, thresholds procedure, exact member protocol and cloud extraction /
inference resource plan before notifying the user to start the GPU.

The 3,045 test-group allocation supports the previously calculated Hoeffding
planning margin of approximately 0.03 for six risk comparisons and the paired
coverage error target 0.05 only under the stated group-independence assumptions.
Neither statistical power nor positive retained coverage is guaranteed. Formal
calibration algorithm and its assumptions must be specified before image access;
these counts do not by themselves calibrate anything.
