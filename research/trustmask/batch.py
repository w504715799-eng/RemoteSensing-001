"""Bounded, journaled study execution; only compact per-ROI statistics persist."""

import fcntl
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from research.trustmask.benefits import evaluate_mask
from research.trustmask.calibration import calibrate, evaluate_comparison, probability
from research.trustmask.pipeline import (
    GRID,
    ROLES,
    Config,
    Observation,
    deploy_mask,
    families,
    group_curves,
    raw_components,
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _write(path, value):
    envelope = {"sha256": _digest(value), "payload": value}
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        stream.write(_json(envelope))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read(path):
    try:
        envelope = json.loads(path.read_text())
        if (
            set(envelope) != {"sha256", "payload"}
            or _digest(envelope["payload"]) != envelope["sha256"]
        ):
            raise ValueError("digest mismatch")
        return envelope["payload"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"corrupt journal or receipt: {path}") from exc


def _persist(path, value):
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"corrupt derived receipt: {path}")
    else:
        _write(path, value)


def _assignments(assignments):
    result, groups, samples = [], set(), set()
    if not isinstance(assignments, list):
        raise ValueError("assignments must be a list")
    for item in assignments:
        if not isinstance(item, dict) or set(item) != {"group_id", "members", "role"}:
            raise ValueError("invalid assignment")
        group, members, role = item["group_id"], item["members"], item["role"]
        if not isinstance(group, str) or not group.strip() or group in groups or role not in ROLES:
            raise ValueError("invalid or duplicate group/role")
        if not isinstance(members, list) or not members:
            raise ValueError("nonempty group members required")
        groups.add(group)
        for sample in members:
            if not isinstance(sample, str) or not sample.strip() or sample in samples:
                raise ValueError("invalid or duplicate sample")
            samples.add(sample)
        result.append(dict(group_id=group, members=sorted(members), role=role))
    if {x["role"] for x in result} != set(ROLES):
        raise ValueError("all five nonempty roles required")
    return sorted(result, key=lambda x: (ROLES.index(x["role"]), x["group_id"]))


def _aggregate(receipts, key):
    grouped = {}
    for receipt in receipts:
        grouped.setdefault(receipt["group_id"], []).append(receipt["summary"][key])
    return np.stack([np.mean(grouped[g], axis=0) for g in sorted(grouped)])


def _config_key(config):
    return _json(asdict(config))


def run_batch(root: Path, assignments: list, provider, *, binding: dict, max_rois: int, alpha=0.05):
    """Process at most max_rois, resuming only the exact bound experiment.

    The provider must authenticate real observations. The journal protects accidental
    corruption and incompatible reuse, not malicious edits by a filesystem owner.
    """
    alpha = probability(alpha, "alpha")
    if type(max_rois) is not int or max_rois < 1:
        raise ValueError("max_rois must be a positive integer")
    if not isinstance(binding, dict) or not binding:
        raise ValueError("nonempty binding required")
    assignments = _assignments(assignments)
    bound = dict(schema="trustmask-batch-v1", assignments=assignments, binding=binding, alpha=alpha)
    digest = _digest(bound)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("batch already has an active writer") from exc
        journal = root / "journal.json"
        if journal.exists():
            if _read(journal) != bound:
                raise ValueError("incompatible resume binding")
        else:
            if any(p.name != ".lock" for p in root.iterdir()):
                raise ValueError("corrupt batch: journal missing in nonempty root")
            _write(journal, bound)
        directory = root / "receipts"
        directory.mkdir(exist_ok=True)
        expected = [
            dict(role=a["role"], group_id=a["group_id"], sample_id=s)
            for a in assignments
            for s in a["members"]
        ]
        receipts = []
        for index, identity in enumerate(expected):
            path = directory / f"{index:08d}.json"
            if not path.exists():
                break
            receipt = _read(path)
            if (
                not isinstance(receipt, dict)
                or any(receipt.get(k) != v for k, v in identity.items())
                or receipt.get("binding_sha256") != digest
                or "summary" not in receipt
            ):
                raise ValueError("corrupt receipt identity or binding")
            receipts.append(receipt)
        pending_temp = directory / f"{len(receipts):08d}.tmp"
        if len(receipts) < len(expected) and pending_temp.is_file():
            pending_temp.unlink()
        if {p.name for p in directory.iterdir()} != {f"{i:08d}.json" for i in range(len(receipts))}:
            raise ValueError("corrupt receipt sequence")
        terminal = root / "terminal_test_started.json"
        frozen_path = root / "frozen_methods.json"
        pretest_count = sum(x["role"] != "test" for x in expected)
        if (terminal.exists() or frozen_path.exists()) and len(receipts) < pretest_count:
            raise ValueError("corrupt batch: frozen methods require complete pretest receipts")
        if terminal.exists() and not frozen_path.exists():
            raise ValueError("corrupt batch: test ledger without frozen methods")
        if any(r["role"] == "test" for r in receipts) and not terminal.exists():
            raise ValueError("corrupt batch: test receipts without started ledger")
        for artifact in (terminal, frozen_path, root / "result.json"):
            if artifact.exists():
                _read(artifact)
        if (root / "result.json").exists() and len(receipts) != len(expected):
            raise ValueError("corrupt batch: result with incomplete receipts")
        processed = 0
        all_configs = [c for configs in families().values() for c in configs]
        unique_configs = {_config_key(c): c for c in all_configs}
        scales, development, selected, frozen = None, {}, {}, {}
        by_role = {r: [] for r in ROLES}
        for receipt in receipts:
            by_role[receipt["role"]].append(receipt)
        for role in ROLES:
            if role == "development_calibration":
                scales = {
                    key: max(
                        float(
                            np.median(
                                np.concatenate([r["summary"][key] for r in by_role["scale_fit"]])
                            )
                        ),
                        1e-8,
                    )
                    for key in by_role["scale_fit"][0]["summary"]
                }
            if role == "development_validation":
                devcal = {
                    key: calibrate(
                        _aggregate(by_role["development_calibration"], key)[:, 0, :],
                        np.array(GRID),
                        alpha=alpha,
                    )
                    for key in unique_configs
                }
            if role == "calibration":
                for name, configs in families().items():
                    candidates = []
                    for index, config in enumerate(configs):
                        key = _config_key(config)
                        summary = _aggregate(by_role["development_validation"], key)
                        candidates.append(
                            dict(
                                index=index,
                                config=asdict(config),
                                validation_risk=float(summary[:, 0, 0].mean()),
                                validation_coverage=float(summary[:, 1, 0].mean()),
                                development_calibration=devcal[key],
                            )
                        )
                    development[name] = candidates
                    feasible = [c for c in candidates if c["validation_risk"] <= alpha]
                    best = (
                        min(
                            feasible,
                            key=lambda c: (
                                -c["validation_coverage"],
                                c["validation_risk"],
                                c["index"],
                            ),
                        )
                        if feasible
                        else candidates[0]
                    )
                    selected[name] = (Config(**best["config"]), not feasible)
            if role == "test":
                for name, (config, force_reject) in selected.items():
                    calibration = calibrate(
                        _aggregate(by_role["calibration"], name)[:, 0, :],
                        np.array(GRID),
                        alpha=alpha,
                    )
                    if force_reject:
                        calibration.update(
                            threshold=-1.0,
                            reject_all=True,
                            calibration_mean_loss=0.0,
                            corrected_calibration_risk=None,
                            rule="structural_rejection_no_development_candidate_qualified",
                            guarantee="structural_zero_loss_from_rejecting_every_pixel",
                        )
                    frozen[name] = dict(
                        config=asdict(config), scales=scales, calibration=calibration
                    )
                _persist(root / "frozen_methods.json", frozen)
                if terminal.exists():
                    _persist(
                        terminal, dict(binding_sha256=digest, frozen_methods_sha256=_digest(frozen))
                    )
            for identity in (x for x in expected if x["role"] == role):
                if any(r["sample_id"] == identity["sample_id"] for r in by_role[role]):
                    continue
                if processed == max_rois:
                    return dict(
                        status="in_progress",
                        processed_rois=processed,
                        completed_rois=len(receipts),
                        total_rois=len(expected),
                        stage=role,
                        binding_sha256=digest,
                    )
                if role == "test":
                    _persist(
                        root / "terminal_test_started.json",
                        dict(binding_sha256=digest, frozen_methods_sha256=_digest(frozen)),
                    )
                row = provider(identity["sample_id"], identity["group_id"])
                if (
                    not isinstance(row, Observation)
                    or row.sample_id != identity["sample_id"]
                    or row.group_id != identity["group_id"]
                ):
                    raise ValueError("provider observation identity mismatch")
                summary = {}
                if role == "scale_fit":
                    for key, values in raw_components(row.features).items():
                        index = np.floor((np.arange(64) + 0.5) * values.size / 64).astype(int)
                        summary[key] = values.ravel()[index].tolist()
                    del values
                else:
                    configs = (
                        unique_configs
                        if role.startswith("development_")
                        else {name: config for name, (config, _) in selected.items()}
                    )
                    for key, config in configs.items():
                        grid = (
                            [devcal[key]["threshold"]]
                            if role == "development_validation"
                            else [frozen[key]["calibration"]["threshold"]]
                            if role == "test"
                            else GRID
                        )
                        _, loss, coverage = group_curves([row], config, scales, grid)
                        summary[key] = [loss[0].tolist(), coverage[0].tolist()]
                        if role == "test":
                            summary[key + ":workload"] = evaluate_mask(
                                row.risk, deploy_mask(row.features, config, scales, grid[0])
                            )
                del row
                receipt = dict(**identity, binding_sha256=digest, summary=summary)
                provenance = getattr(provider, "last_provenance", None)
                if isinstance(provenance, dict):
                    receipt["provenance"] = json.loads(_json(provenance))
                _write(directory / f"{len(receipts):08d}.json", receipt)
                receipts.append(receipt)
                by_role[role].append(receipt)
                processed += 1
        _persist(
            root / "terminal_test_started.json",
            dict(binding_sha256=digest, frozen_methods_sha256=_digest(frozen)),
        )
        losses, coverages, workload = {}, {}, {}
        keys = (
            "accepted_pixels",
            "review_pixels",
            "review_tiles",
            "total_tiles",
            "accepted_high_error_pixels",
            "high_error_pixels",
        )
        for name in frozen:
            aggregate = _aggregate(by_role["test"], name)
            losses[name], coverages[name] = aggregate[:, 0, 0], aggregate[:, 1, 0]
            summaries = [r["summary"][name + ":workload"] for r in by_role["test"]]
            workload[name] = {key: sum(s[key] for s in summaries) for key in keys}
            workload[name].update(
                all_rejected_rois=sum(s["all_rejected"] for s in summaries),
                physical_area_m2=None,
                human_time_measured=False,
            )
        result = dict(
            schema="trustmask-integrated-study-v1",
            evidence_kind=binding.get(
                "evidence_kind", "offline_precomputed_inputs_not_provenance_authenticated"
            ),
            frozen_methods=frozen,
            frozen_methods_sha256=_digest(frozen),
            development=development,
            test_group_ids=sorted({r["group_id"] for r in by_role["test"]}),
            evaluation=evaluate_comparison(losses, coverages, alpha=alpha),
            workload=workload,
            stage_counts={
                role: dict(rois=len(rows), groups=len({r["group_id"] for r in rows}))
                for role, rows in by_role.items()
            },
            weighting="equal_groups_then_equal_rois",
            method_family_fixed_before_calibration=True,
            threshold_grid=list(GRID),
            scale_fit_rule="64_fixed_midpoint_positions_per_roi_median_floor_1e-8",
            test_group_observations={
                name: dict(loss=losses[name].tolist(), coverage=coverages[name].tolist())
                for name in frozen
            },
        )
        if "evidence_kind" in binding:
            result["binding_sha256"] = digest
        _persist(root / "result.json", result)
        return dict(
            status="complete",
            processed_rois=processed,
            completed_rois=len(receipts),
            total_rois=len(expected),
            stage="complete",
            binding_sha256=digest,
            result=_read(root / "result.json"),
        )
