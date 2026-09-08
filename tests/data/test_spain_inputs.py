import numpy as np
import pandas as pd
import pytest
import torch

from trustsr.data import spain_inputs


def members():
    return [
        dict(
            roi=f"ROI_{i}",
            lr_file=f"lr{i}",
            hr_file=f"hr{i}",
            lr_gee_id="scene",
            crs="EPSG:32630",
            affine=[2.5, 0, 10 * i, 0, -2.5, 20],
        )
        for i in (1, 2)
    ]


def test_member_binding_uses_tensor_row_position_not_sorted_order_or_dataframe_index():
    expected = members()
    embedded = pd.DataFrame(expected[::-1], index=[8, 8])
    embedded["quality"] = ["never-used", "never-used"]
    embedded["affine"] = [",".join(map(str, v)) for v in embedded["affine"]]
    assert spain_inputs.bind_member_rows(embedded, expected) == (1, 0)


@pytest.mark.parametrize("fault", ["duplicate", "asset", "affine", "missing", "expected_duplicate"])
def test_member_mismatch_cannot_silently_bind_a_different_image(fault):
    expected = members()
    embedded = pd.DataFrame(members())
    if fault == "duplicate":
        embedded.loc[1, "roi"] = "ROI_1"
    if fault == "asset":
        embedded.loc[0, "hr_file"] = "different-image"
    if fault == "affine":
        embedded.at[0, "affine"] = [2.5, 0, 999, 0, -2.5, 20]
    if fault == "missing":
        embedded = embedded.drop(columns=["lr_gee_id"])
    if fault == "expected_duplicate":
        expected[1] = expected[0]
    with pytest.raises(ValueError):
        spain_inputs.bind_member_rows(embedded, expected)


def arrays():
    lr = np.zeros((12, 128, 128), dtype=np.uint16)
    hr = np.zeros((4, 512, 512), dtype=np.uint16)
    return lr, hr


def test_rgbn_order_full_grid_and_saturation_counts():
    lr, hr = arrays()
    for index, value in zip((3, 2, 1, 7), (1000, 2000, 3000, 12000), strict=True):
        lr[index] = value
    hr[:] = np.array([4000, 5000, 6000, 15000], dtype=np.uint16)[:, None, None]
    before = (lr.copy(), hr.copy())
    pair, receipt = spain_inputs.prepare_spain_pair("ROI_00001", lr, hr)
    torch.testing.assert_close(pair.lr[:, 0, 0], torch.tensor([0.1, 0.2, 0.3, 1.0]))
    torch.testing.assert_close(pair.hr[:, 0, 0], torch.tensor([0.4, 0.5, 0.6, 1.0]))
    assert pair.lr.shape == (4, 128, 128) and pair.hr.shape == (4, 512, 512)
    assert receipt == {
        "lr_clipped_by_band": [0, 0, 0, 16384],
        "hr_clipped_by_band": [0, 0, 0, 262144],
    }
    assert np.array_equal(lr, before[0]) and np.array_equal(hr, before[1])
    pair.validate()


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), 65535.0, 32768.0])
@pytest.mark.parametrize("target", ["lr", "hr"])
def test_invalid_selected_pixels_are_rejected_not_repaired(value, target):
    lr, hr = [a.astype(np.float32) for a in arrays()]
    if target == "lr":
        lr[3, 0, 0] = value
    else:
        hr[0, 0, 0] = value
    with pytest.raises(ValueError):
        spain_inputs.prepare_spain_pair("ROI_00001", lr, hr)


@pytest.mark.parametrize("fault", ["lr_shape", "hr_shape", "object", "complex", "name"])
def test_bad_decoded_contract_is_rejected(fault):
    lr, hr = arrays()
    name = "ROI_00001"
    if fault == "lr_shape":
        lr = lr[:4]
    if fault == "hr_shape":
        hr = hr[:, :-1]
    if fault == "object":
        lr = lr.astype(object)
    if fault == "complex":
        hr = hr.astype(np.complex64)
    if fault == "name":
        name = ""
    with pytest.raises(ValueError):
        spain_inputs.prepare_spain_pair(name, lr, hr)


def test_zero_is_preserved_without_inventing_a_nodata_mask():
    pair, receipt = spain_inputs.prepare_spain_pair("ROI_00001", *arrays())
    assert torch.count_nonzero(pair.lr) == 0 and torch.count_nonzero(pair.hr) == 0
    assert sum(receipt["lr_clipped_by_band"]) == sum(receipt["hr_clipped_by_band"]) == 0
