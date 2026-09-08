# SEN2NAIPv2 crosssensor provenance request — unsent draft

Status: local draft only; no discussion, issue, email or other message has been sent.
Suggested destination: the official SEN2NAIPv2 dataset discussion forum.

**Title:** Per-member Sentinel-2 source metadata for SEN2NAIPv2 crosssensor

Hello, we are designing a source-aware geographic split for a study using the
SEN2NAIPv2 crosssensor collection. To prevent LR source sharing across development,
calibration and evaluation, we need the original Sentinel-2 provenance for each ROI.

Our pinned source is `tacofoundation/SEN2NAIPv2`, revision
`c370504201072fdb1dd388013ab8c0fc7d00a57e`, file `sen2naipv2-crosssensor.taco`
(8,000 members). The inspected top directory and three sampled nested directories
contain member IDs, geometry and timestamps but no explicit Sentinel-2 product ID.

Is there an existing metadata-only manifest or construction script providing:

- TACO member ID;
- original Sentinel-2 product/granule or GEE image ID;
- every contributing source ID if LR was mosaicked or composited;
- original NAIP image ID;
- actual LR and HR acquisition times;
- ROI geometry/CRS and the dataset revision the mapping applies to?

We recovered our original metadata manifest and verified that top-level
`stac:time_start` equals the nested LR timestamp for all 8,000 members. Both LR
and HR timestamps fall exactly at midnight when interpreted in Europe/Madrid;
all HR dates in that interpretation equal the NAIP filename date. Was a date-only
value serialized via a local-time datetime, and what timezone/date convention
should consumers use? We are not assuming this identifies your build timezone.
How would a mosaic's time and contributing products be represented?

For example, one member is
`NA5120_E1183N0757__m_3912321_nw_10_060_20220710`.
Its stored LR timestamp is `2022-07-10T22:00:00Z`. A bounded public L2A catalog
query covering July 10–11 returns candidates on July 11 in tiles 10SDJ and 10TDK.
Can the construction record identify the actual tile(s) and processing version?
For `NA5120_E1185N0724__m_3812242_ne_10_060_20200602`, our bounded query returns
June 3, 2020 tile 10SEH candidates with processing baselines 02.14 and 05.00.
We retain these as candidates only, not verified mappings.

We also inspected three metadata files in the original SEN2NAIP ZIP: those have
`s2_id`, but we could not establish their mapping to this v2 collection. A version
mapping, if one exists, would be helpful too.

We only need metadata or a link to the construction records, not image downloads
or model weights. Thank you.
