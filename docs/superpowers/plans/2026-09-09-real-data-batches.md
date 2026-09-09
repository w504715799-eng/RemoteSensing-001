# Real-data batch execution

User-authorized continuation on main. Public official release is sufficient; no
per-member Sentinel source mapping or new provenance inquiry. Old A/B/C pixels and
prediction caches remain closed. No cloud/GPU execution during local development.

## Design and tasks

1. `research/trustmask/real_data.py`: bounded selected-member TACO reads, pinned
   top metadata validation, fixed RGBN support and radiometry, lazy verified LDSR
   five-seed inference, seed3407 full-grid R9 and reference-free features. Tests
   reject unexpected members, malformed source metadata and invalid support.
2. `research/trustmask/batch.py`: one ROI resident; compact atomic receipts,
   exclusive writer lock and binding validation; five ordered roles. Freeze scales,
   then all development choices, then all formal thresholds, before test access.
   A separate terminal ledger allows only same-protocol crash continuation.
   Tests compare against the in-memory reference, vary batch boundaries, inject
   interruption/corruption, and verify no repeated completed ROI inference.
3. `research/trustmask/batch_cli.py`: exact assignment digest/count validation,
   protocol with implementation and model binding, metadata-only preflight and
   explicit bounded run command. CLI tests verify tampering fails before provider.
4. Document persistent storage commands, dependencies, resume and scientific limits;
   review components and combined integration; run scoped tests, frozen protocol
   and staged-data gates, then commit. Notify user when GPU execution is ready.

## Rulings

- Use the existing preview's exact 7,477 members / 6,158 groups. Freeze it for this
  new study without rewriting the historical preview or old terminal ledger.
- Pin source size and known metadata digests, and validate selected assets. The
  published full-object SHA is declared provenance, not a runtime claim that every
  container byte was reread. No old pixels are opened to hash the whole container.
- Keep only compact ROI summaries on disk. Retry an interrupted ROI under the same
  frozen binding, record terminal start before test inference, and never rerun
  completed receipts. This is an operational ledger, not protection against an
  operator deliberately deleting/copying study directories.
- Statistical conclusions remain conditional on exchangeability/independence of
  the geographic groups; public availability does not establish these assumptions.
