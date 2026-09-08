"""Nominal footprint geometry from metadata, not pixel validity or independence proof."""

import math
import re
from numbers import Real

import numpy as np
from rasterio.warp import transform as transform_coordinates


def _values(value, length):
    if (not isinstance(value, (list, tuple)) or len(value) != length
            or any(isinstance(v, bool) or not isinstance(v, Real)
                   or not math.isfinite(v) for v in value)):
        raise ValueError('finite numeric metadata vector required')
    return tuple(float(v) for v in value)


def _distances_m(lon, lat, origin):
    x, y = np.radians(lon), np.radians(lat)
    a, b = np.radians(origin)
    hav = np.sin((y - b) / 2) ** 2 + np.cos(y) * np.cos(b) * np.sin((x - a) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))


def footprint_summary(crs, geotransform, shape, centroid):
    """Inspect fixed crosssensor520²UTM footprint and a sampled geodesic perimeter.

    Radius is a spherical, sampled geometry diagnostic, not a certified enclosure
    or exact polygon-to-polygon separation. Geotransform uses GDAL coefficient order.
    """
    if not isinstance(crs, str) or not re.fullmatch(r'EPSG:326(?:0[1-9]|[1-5][0-9]|60)', crs):
        raise ValueError('reviewed UTM northern CRS required')
    x, dx, rx, y, ry, dy = _values(geotransform, 6)
    if (dx, rx, ry, dy) != (2.5, 0., 0., -2.5):
        raise ValueError('north-up2.5m grid required')
    if not isinstance(shape, (list, tuple)) or list(shape) != [520, 520] or any(
            type(n) is not int for n in shape):
        raise ValueError('520x520 metadata grid required')
    center = _values(centroid, 2)
    if abs(center[0]) > 180 or abs(center[1]) > 90:
        raise ValueError('invalid geographic centroid')
    t = np.linspace(0, 520, 17)
    cols = np.concatenate((t, np.full(17, 520), t[::-1], np.zeros(17)))
    rows = np.concatenate((np.zeros(17), t, np.full(17, 520), t[::-1]))
    xs = (x + cols * dx).tolist() + [x + 260 * dx]
    ys = (y + rows * dy).tolist() + [y + 260 * dy]
    lon, lat = transform_coordinates(crs, 'EPSG:4326', xs, ys)
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError('nonfinite transformed footprint')
    distances = _distances_m(lon, lat, center)
    return dict(
        nominal_width_m=1300., nominal_height_m=1300., nominal_area_m2=1690000.,
        centroid_discrepancy_m=float(distances[-1]),
        sampled_perimeter_radius_m=float(max(distances[:-1])),
        lon_min=float(min(lon[:-1])), lon_max=float(max(lon[:-1])),
        lat_min=float(min(lat[:-1])), lat_max=float(max(lat[:-1])),
        valid_pixel_support_verified=False,
    )
