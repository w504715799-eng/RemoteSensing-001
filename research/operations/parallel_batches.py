"""Audited process-level scheduling amendment for the unchanged frozen study."""

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from research.trustmask.batch import _digest, _persist, _read, run_batch
from research.trustmask.batch_cli import runtime_binding, verify_protocol
from research.trustmask.real_data import LocalTacoProvider

_WORKER = None


def initialize_worker(taco, members, models, device, fingerprint):
    global _WORKER
    _WORKER = LocalTacoProvider(taco, members, models, device=device)
    if _WORKER.preflight()["source_file_fingerprint"] != fingerprint:
        raise ValueError("worker source fingerprint changed")


def compute(sample, group):
    observation = _WORKER(sample, group)
    return observation, _WORKER.last_provenance


class OrderedProvider:
    """Submit at most workers jobs, strictly within the currently requested role."""

    def __init__(self, expected, submit, *, workers, guard):
        if type(workers) is not int or not 1 <= workers <= 2:
            raise ValueError("workers must be one or two")
        self.expected = tuple(expected)
        self.submit, self.workers, self.guard = submit, workers, guard
        self.cursor = self.scheduled = 0
        self.pending = {}
        self.role = None
        self.last_provenance = None

    def __call__(self, sample, group):
        if self.cursor >= len(self.expected) or self.expected[self.cursor][1:] != (group, sample):
            raise ValueError("provider request outside exact remaining order")
        role = self.expected[self.cursor][0]
        if role != self.role:
            if self.pending:
                raise ValueError("pending work crossed a role barrier")
            self.guard(role)
            self.role = role
        while self.scheduled < len(self.expected) and len(self.pending) < self.workers:
            next_role, next_group, next_sample = self.expected[self.scheduled]
            if next_role != role:
                break
            self.pending[self.scheduled] = self.submit(next_sample, next_group)
            self.scheduled += 1
        row, provenance = self.pending.pop(self.cursor).result()
        self.cursor += 1
        self.last_provenance = provenance
        return row


def terminal_guard(root, digest, role):
    if role == "test":
        frozen = _read(root / "frozen_methods.json")
        ledger = _read(root / "terminal_test_started.json")
        if ledger != dict(binding_sha256=digest, frozen_methods_sha256=_digest(frozen)):
            raise ValueError("terminal ledger does not bind frozen methods")


def interrupted(signum, frame):
    raise KeyboardInterrupt("operational stop; preserve committed receipts")


def main(argv=None):
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "assignments", "taco", "model-dir", "study-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-rois", type=int, default=7477)
    args = parser.parse_args(argv)
    if args.max_rois < 1:
        parser.error("positive ROI budget required")
    protocol, assignments = verify_protocol(args.protocol, args.assignments)
    root = args.study_dir
    # No initialization of a different study: this wrapper only resumes a verified journal.
    journal = _read(root / "journal.json")
    members = {s: a["group_id"] for a in assignments for s in a["members"]}
    with LocalTacoProvider(args.taco, members, args.model_dir, device=args.device) as preflight:
        fingerprint = preflight.preflight()["source_file_fingerprint"]
    binding = dict(
        protocol=protocol,
        runtime=runtime_binding(),
        device=args.device,
        source_file=fingerprint,
        evidence_kind="real_public_source_selected_assets",
    )
    if journal["binding"] != binding or journal["alpha"] != protocol["alpha"]:
        raise ValueError("execution amendment cannot change scientific/runtime binding")
    with (root / ".operations.lock").open("a") as operation_lock:
        fcntl.flock(operation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Establish the cutover prefix only while the existing writer is absent.
        with (root / ".lock").open("a") as writer_lock:
            fcntl.flock(writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            files = sorted((root / "receipts").glob("*.json"))
            prefix = []
            for i, path in enumerate(files):
                if path.name != f"{i:08d}.json":
                    raise ValueError("receipt prefix is not contiguous")
                _read(path)
                prefix.append(hashlib.sha256(path.read_bytes()).hexdigest())
            expected = [
                (a["role"], a["group_id"], s) for a in journal["assignments"] for s in a["members"]
            ]
            if len(files) > len(expected):
                raise ValueError("too many existing receipts")
            amendment_dir = root / "execution_amendments"
            amendment_dir.mkdir(exist_ok=True)
            amendment = dict(
                schema="trustmask-process-scheduling-amendment-v1",
                base_binding_sha256=_digest(journal),
                scientific_protocol_sha256=protocol["protocol_sha256"],
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                workers=args.workers,
                max_rois=args.max_rois,
                start_index=len(files),
                existing_receipt_sha256=prefix,
                scheduling="ordered_commit_same_role_prefetch_spawn_independent_rng",
                memory_policy="at_most_two_outstanding_roi_jobs_plus_coordinator_summary",
                original_one_roi_memory_policy_amended=True,
                scientific_definition_changed=False,
                authorized_by="user_parallel_acceleration_request_2026-09-09",
            )
            _persist(amendment_dir / f"{len(files):08d}-{_digest(amendment)[:16]}.json", amendment)
        remaining = expected[len(files) : len(files) + args.max_rois]
        started = time.time()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(args.taco, members, args.model_dir, args.device, fingerprint),
        ) as pool:
            provider = OrderedProvider(
                remaining,
                lambda sample, group: pool.submit(compute, sample, group),
                workers=args.workers,
                guard=lambda role: terminal_guard(root, _digest(journal), role),
            )
            result = run_batch(
                root,
                assignments,
                provider,
                binding=binding,
                max_rois=args.max_rois,
                alpha=protocol["alpha"],
            )
        result.pop("result", None)
        result.update(wall_seconds=time.time() - started, workers=args.workers, pid=os.getpid())
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
