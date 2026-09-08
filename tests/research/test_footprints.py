import pytest

from research.trustmask.footprints import footprint_summary


def test_nominal_square_area_and_center_at_utm_central_meridian():
    transform = [499350., 2.5, 0., 650., 0., -2.5]
    result = footprint_summary('EPSG:32631', transform, [520, 520], [3., 0.])
    assert result['nominal_width_m'] == 1300
    assert result['nominal_height_m'] == 1300
    assert result['nominal_area_m2'] == 1690000
    assert result['centroid_discrepancy_m'] < 1e-6
    assert 900 < result['sampled_perimeter_radius_m'] < 940
    assert result['valid_pixel_support_verified'] is False
    assert transform == [499350., 2.5, 0., 650., 0., -2.5]


def test_report_centroid_mismatch_instead_of_hiding_it():
    r = footprint_summary('EPSG:32631', [499350., 2.5, 0., 650., 0., -2.5],
                          [520, 520], [4., 0.])
    assert r['centroid_discrepancy_m'] > 100000


@pytest.mark.parametrize('crs,transform,shape,centroid', [
    ('EPSG:4326', [0., 2.5, 0., 0., 0., -2.5], [520, 520], [0, 0]),
    ('EPSG:32631', [0., 2.5, 1., 0., 0., -2.5], [520, 520], [0, 0]),
    ('EPSG:32631', [0., 10., 0., 0., 0., -10.], [520, 520], [0, 0]),
    ('EPSG:32631', [0., 2.5, 0., 0., 0., -2.5], [130, 130], [0, 0]),
    ('EPSG:32631', [float('nan'), 2.5, 0., 0., 0., -2.5], [520, 520], [0, 0]),
    ('EPSG:32631', [0., 2.5, 0., 0., 0., -2.5], [520, 520], [181, 0]),
])
def test_unreviewed_geometry_is_rejected(crs, transform, shape, centroid):
    with pytest.raises(ValueError):
        footprint_summary(crs, transform, shape, centroid)
