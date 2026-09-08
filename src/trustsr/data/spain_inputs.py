"""Strict preparation of already-decoded Spain arrays; no dataset access."""

import numpy as np
import pandas as pd
import torch

from trustsr.contracts import SRPair


def bind_member_rows(embedded: pd.DataFrame, expected: list[dict]) -> tuple[int, ...]:
    """Bind a pinned projected member list to positional tensor rows, ignoring quality."""
    fields = ("roi", "lr_file", "hr_file", "lr_gee_id", "crs", "affine")
    if not isinstance(embedded, pd.DataFrame) or not embedded.columns.is_unique:
        raise ValueError("embedded metadata must have unique columns")
    if not set(fields).issubset(embedded.columns) or len(embedded) != len(expected) or not expected:
        raise ValueError("member count or fields mismatch")
    ids = [m["roi"] for m in expected]
    if len(set(ids)) != len(ids):
        raise ValueError("expected members must be unique")
    rows = embedded.loc[:, list(fields)].to_dict(orient="records")
    by_id = {row["roi"]: (position, row) for position, row in enumerate(rows)}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError("embedded member identities mismatch")
    positions = []
    for member in expected:
        position, row = by_id[member["roi"]]
        for field in fields[:-1]:
            if row[field] != member[field]:
                raise ValueError("member source metadata mismatch")
        affine = row["affine"]
        try:
            values = np.asarray(
                affine.split(",") if isinstance(affine, str) else affine, dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid embedded affine") from exc
        if (
            values.shape != (6,)
            or not np.isfinite(values).all()
            or not np.array_equal(values, member["affine"])
        ):
            raise ValueError("member affine mismatch")
        positions.append(position)
    return tuple(positions)


def prepare_spain_pair(sample_id: str, lr_l2a: np.ndarray, hr_harmonized: np.ndarray):
    """Select RGBN and apply the proposed study normalization, without mutation.

    Caller must independently authenticate the package and bind this row to its member.
    Numeric checks do not establish source nodata semantics.
    """
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a nonempty string")
    for value, shape in ((lr_l2a, (12, 128, 128)), (hr_harmonized, (4, 512, 512))):
        if type(value) is not np.ndarray or value.shape != shape:
            raise ValueError("expected exact benchmark ndarray dimensions")
        if value.dtype.kind not in "uif":
            raise ValueError("expected integer or floating reflectance-DN arrays")
    outputs, counts = [], []
    for value in (lr_l2a[[3, 2, 1, 7]], hr_harmonized):
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 32767):
            raise ValueError("invalid reflectance-DN, including nonfinite or nodata sentinel")
        counts.append(np.count_nonzero(value > 10000, axis=(1, 2)).tolist())
        normalized = np.minimum(value.astype(np.float64), 10000) / 10000
        outputs.append(torch.from_numpy(normalized.astype(np.float32)))
    pair = SRPair(sample_id, "opensr-test/spain-v2-decoded", outputs[0], outputs[1], 4)
    pair.validate()
    return pair, dict(lr_clipped_by_band=counts[0], hr_clipped_by_band=counts[1])
