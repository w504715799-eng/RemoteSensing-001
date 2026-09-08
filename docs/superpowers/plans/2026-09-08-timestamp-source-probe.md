# Bounded timestamp/source probe

Continue the approved metadata-only provenance investigation on main. No GPU,
image/asset reads, old prediction caches, external messages or split changes.

1. Replay the pinned recovered projection. Quantify whether LR/HR timestamps
   correspond to midnight in Europe/Madrid and whether HR calendar dates then
   equal the NAIP ID suffix. This is a consistency hypothesis, not evidence of
   the actual upstream host timezone. Inspect available upstream writer code
   for a plausible mechanism; distinguish generic implementation from build proof.
2. Before viewing catalog results, retain the exact three previously declared
   members in `crosssensor-source-footprint-audit-v1.json` / `nested_probes`.
   For each query Earth Search `sentinel-2-l2a` metadata only. Use the union of
   the stored LR UTC calendar day and its following day (a fixed 48-hour window),
   covering the raw-date and inferred local-date interpretations. Search a
   centroid-centered +/-0.02 degree bounding box as a discovery envelope, not
   a verified footprint certificate. No cloud/quality filters or ranking.
3. At most three successful query responses, 50 results and 1 MiB per response;
   at most two attempts per query for transport failures. Request no assets;
   never follow asset or pagination links. Reject pagination/overflow as incomplete.
   Keep all returned candidate identities/times/tile/orbit/product metadata. Do
   not turn a unique catalog result into an asserted source mapping. Record
   raw response hashes, request parameters and offline projected results.
4. Publish timestamp aggregates and bounded candidate evidence, run focused
   synthetic tests and independent review, update handoff with the next concrete
   dependency. Source-product construction/compositing provenance remains a
   separate gate before grouping or experiment execution.

Execution note after first response: the service reports matched=returned=2 but
also supplies a next link. Preserve the response and its candidates as first-page
evidence, explicitly mark enumeration uncertified, and do not follow that link.
The strict projection mode still rejects incomplete pagination. The reporting
runner can retain partial evidence with an explicit status; it never upgrades
the first page to complete enumeration. Query scope and request budget are unchanged.
