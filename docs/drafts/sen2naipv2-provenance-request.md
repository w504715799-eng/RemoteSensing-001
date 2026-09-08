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

Could you also clarify the meaning of top-level `stac:time_start` and whether it
always equals the LR acquisition time, including how mosaics would be represented?

For example, one member is
`NA5120_E1183N0757__m_3912321_nw_10_060_20220710`.
We also inspected three metadata files in the original SEN2NAIP ZIP: those have
`s2_id`, but we could not establish their mapping to this v2 collection. A version
mapping, if one exists, would be helpful too.

We only need metadata or a link to the construction records, not image downloads
or model weights. Thank you.
