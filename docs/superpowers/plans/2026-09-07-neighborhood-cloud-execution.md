# Development neighborhood CPU execution

Scope approved: cloud execution only, development 120 ROIs, fixed K5 seeds 3407–3411.
No B/C pixel/cache access, no external test, no inference, no raw downloads.

1. Test a development membership boundary, exact K5 selection, file hash rejection, and the
   preregistered mean-R9/tie selection rule before implementing the runner.
2. Pin published A result/audit; match development membership to the frozen manifest before
   pixel access. Only read its development asset paths and exact audited K5 filenames.
3. Verify all 600 cache sidecars/tensors and 240 asset file hashes before scoring. Reuse the
   existing pair loader, verify normalized LR/HR and R1/R9 hashes against published A.
4. Compute windows 3 and 9 with sigma 1, CPU float64, one sequential worker. Record every ROI
   and both candidates; never change the candidates based on results. Recompute K5 AURC as a
   numerical compatibility check. Output new files only, never overwrite historical results.
5. Commit tested local implementation; deploy an isolated copy of that exact commit to cloud.
   Run synthetic tests there, then compute and repeat CPU evaluation to verify science fields.
6. Persist small result and runtime files on durable storage and recheck hashes. Only scientific
   results may return to Git. Report actual CPU timing, no fabricated GPU estimate.
7. Notify user when this cloud job is complete and server may be stopped. This does not mean
   the external protocol or full paper is complete.
