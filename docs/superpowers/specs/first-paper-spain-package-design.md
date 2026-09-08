# Spain package decoder implementation contract

2026-09-08. Implements the authorized Task 5 safe-loading gate; no external access or freeze.

`spain_package.decode_package(path, expected_sha256, expected_size, members)` is a local,
no-download primitive. The future frozen runner supplies pinned package identities from the
metadata audit, not user-selectable replacement identities. Read a regular file with a bounded
read, authenticate the exact bytes before starting a decoder, and pass those same bytes to it.

A separate Python process imports only NumPy/Pandas plus the standard library. It has a wall
 timeout, CPU/address-space/file-size limits, one numerical-library thread, and a restricted
Unpickler with exact NumPy/Pandas reconstruction globals. Reject extension-registry and persistent
reference opcodes, unknown globals, trailing pickle data, and unexpected output schemas. Never
fall back to unrestricted pickle or dynamically expand the allowlist on external data.
This is defense in depth for the pinned upstream package, not a general hostile-pickle sandbox:
NumPy/Pandas native code remains trusted, and process resource limits are not OS isolation.

The child exports only numeric L2A/HRharm arrays and six projected metadata fields as NPZ+JSON,
without object arrays. The parent reads with `allow_pickle=False`, uses existing positional member
binding, and prepares full-grid pairs with the existing normalization contract. Extra upstream
image keys and quality columns are neither exported nor used. Check array types, exact row counts,
full grids, finite selected data and range before returning any pair. A package/member contract
error stops the entire call. Unknown source nodata semantics remain a disclosed freeze gate.

Tests use generated NumPy/Pandas packages only: row reversal, unexpected members/shapes/object
arrays, wrong identity before decoder entry, malicious reduce, extension opcodes, trailing streams.
Run only package/input tests and changed-file lint; no all-repository test run for this checkpoint.
