from datetime import datetime, timedelta

import numpy as np

import sharppy.sharptab.profile as profile
import sharppy.sharptab.prof_collection as prof_collection
import sharppy.sharptab.thermo as thermo
import sharppy.sharptab.utils as utils

try:
    import xarray as xr
except ImportError:
    xr = None


ERA5_STORE_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_VARIABLES = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "surface_pressure",
    "geopotential_at_surface",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
GRAVITY = 9.80665

_ERA5_DATASET = None


def build_profile_collection(point, run_dt, forecast_hours):
    if xr is None:
        raise ImportError("ERA5 point-forecast support requires the 'xarray' package.")

    profiles = []
    dates = []
    for fhour in forecast_hours:
        valid_dt = run_dt + timedelta(hours=fhour)
        sample = _sample_point(point, valid_dt)
        profiles.append(_build_profile_from_sample(point, valid_dt, sample))
        dates.append(valid_dt)

    collection = prof_collection.ProfCollection({"": profiles}, dates)
    collection.setMeta("loc", _point_label(point))
    collection.setMeta("observed", False)
    collection.setMeta("base_time", run_dt)
    return collection


def get_latest_time():
    ds = _get_dataset()
    latest_str = ds.attrs.get("valid_time_stop_era5t") or ds.attrs.get("valid_time_stop")
    latest_date = datetime.strptime(latest_str, "%Y-%m-%d")
    return latest_date.replace(hour=23)


def _get_dataset():
    global _ERA5_DATASET
    if _ERA5_DATASET is None:
        _ERA5_DATASET = xr.open_zarr(
            ERA5_STORE_URL,
            chunks=None,
            storage_options={"token": "anon"},
        )
    return _ERA5_DATASET


def _sample_point(point, valid_dt):
    ds = _get_dataset()
    valid_dt = valid_dt.replace(minute=0, second=0, microsecond=0)
    time_value = np.datetime64(valid_dt)

    lat = float(point["lat"])
    lon = float(point["lon"]) % 360.0

    lat_vals = np.asarray(ds.latitude.values, dtype=float)
    lon_vals = np.asarray(ds.longitude.values, dtype=float)
    i0, i1, yweight = _bounding_indices(lat_vals, lat)
    j0, j1, xweight = _bounding_indices_cyclic(lon_vals, lon)

    subset = ds[ERA5_VARIABLES].sel(time=time_value).isel(
        latitude=[i0, i1],
        longitude=[j0, j1],
    )

    return {
        "levels": np.asarray(subset.level.values, dtype=float),
        "t": _bilinear(np.asarray(subset["temperature"].values, dtype=float), xweight, yweight),
        "q": _bilinear(np.asarray(subset["specific_humidity"].values, dtype=float), xweight, yweight),
        "u": _bilinear(np.asarray(subset["u_component_of_wind"].values, dtype=float), xweight, yweight),
        "v": _bilinear(np.asarray(subset["v_component_of_wind"].values, dtype=float), xweight, yweight),
        "z": _bilinear(np.asarray(subset["geopotential"].values, dtype=float), xweight, yweight),
        "sp": float(_bilinear(np.asarray(subset["surface_pressure"].values, dtype=float), xweight, yweight)),
        "zs": float(_bilinear(np.asarray(subset["geopotential_at_surface"].values, dtype=float), xweight, yweight)),
        "t2m": float(_bilinear(np.asarray(subset["2m_temperature"].values, dtype=float), xweight, yweight)),
        "d2m": float(_bilinear(np.asarray(subset["2m_dewpoint_temperature"].values, dtype=float), xweight, yweight)),
        "u10": float(_bilinear(np.asarray(subset["10m_u_component_of_wind"].values, dtype=float), xweight, yweight)),
        "v10": float(_bilinear(np.asarray(subset["10m_v_component_of_wind"].values, dtype=float), xweight, yweight)),
    }


def _build_profile_from_sample(point, valid_dt, sample):
    surface_pres = float(sample["sp"]) * 0.01
    surface_hght = float(sample["zs"]) / GRAVITY
    surface_tmpc = thermo.ktoc(float(sample["t2m"]))
    surface_dwpc = thermo.ktoc(float(sample["d2m"]))
    surface_u = utils.MS2KTS(float(sample["u10"]))
    surface_v = utils.MS2KTS(float(sample["v10"]))

    pres = np.asarray(sample["levels"], dtype=float)
    hght = np.asarray(sample["z"], dtype=float) / GRAVITY
    tmpc = thermo.ktoc(np.asarray(sample["t"], dtype=float))
    u = utils.MS2KTS(np.asarray(sample["u"], dtype=float))
    v = utils.MS2KTS(np.asarray(sample["v"], dtype=float))

    q = np.clip(np.asarray(sample["q"], dtype=float), 1.0e-10, 0.999999)
    mix_ratio = q / np.clip(1.0 - q, 1.0e-10, None)
    vapr = (pres * mix_ratio) / (0.62197 + mix_ratio)
    dwpc = thermo.temp_at_vappres(np.clip(vapr, 1.0e-6, None))

    above_ground = pres < (surface_pres - 0.1)
    pres = pres[above_ground]
    hght = hght[above_ground]
    tmpc = tmpc[above_ground]
    dwpc = dwpc[above_ground]
    u = u[above_ground]
    v = v[above_ground]

    valid = (
        np.isfinite(pres)
        & np.isfinite(hght)
        & np.isfinite(tmpc)
        & np.isfinite(dwpc)
        & np.isfinite(u)
        & np.isfinite(v)
        & (pres > 0)
    )
    pres = pres[valid]
    hght = hght[valid]
    tmpc = tmpc[valid]
    dwpc = dwpc[valid]
    u = u[valid]
    v = v[valid]

    pres = np.concatenate(([surface_pres], pres))
    hght = np.concatenate(([surface_hght], hght))
    tmpc = np.concatenate(([surface_tmpc], tmpc))
    dwpc = np.concatenate(([surface_dwpc], dwpc))
    u = np.concatenate(([surface_u], u))
    v = np.concatenate(([surface_v], v))

    order = np.argsort(pres)[::-1]
    prof = profile.create_profile(
        profile="raw",
        pres=pres[order],
        hght=hght[order],
        tmpc=tmpc[order],
        dwpc=dwpc[order],
        u=u[order],
        v=v[order],
        location=_point_label(point),
        date=valid_dt,
        latitude=float(point["lat"]),
        missing=-9999.0,
        strictQC=False,
    )
    return prof


def _point_label(point):
    return "%.3f, %.3f" % (float(point["lat"]), float(point["lon"]))


def _bounding_indices(values, target):
    vals = np.asarray(values, dtype=float)
    ascending = vals[0] < vals[-1]
    work = vals if ascending else vals[::-1]
    idx = np.searchsorted(work, target)
    idx = min(max(idx, 1), len(work) - 1)
    low = idx - 1
    high = idx

    if ascending:
        i0, i1 = low, high
        v0, v1 = vals[i0], vals[i1]
    else:
        i0 = len(vals) - 1 - high
        i1 = len(vals) - 1 - low
        v0, v1 = vals[i0], vals[i1]

    if v1 == v0:
        return i0, i1, 0.0
    weight = (target - v0) / (v1 - v0)
    return i0, i1, min(max(weight, 0.0), 1.0)


def _bounding_indices_cyclic(values, target):
    vals = np.asarray(values, dtype=float)
    idx = np.searchsorted(vals, target)

    if idx == 0:
        j0 = len(vals) - 1
        j1 = 0
        v0 = vals[j0] - 360.0
        v1 = vals[j1]
    elif idx >= len(vals):
        j0 = len(vals) - 1
        j1 = 0
        v0 = vals[j0]
        v1 = vals[j1] + 360.0
    else:
        j0 = idx - 1
        j1 = idx
        v0 = vals[j0]
        v1 = vals[j1]

    if v1 == v0:
        return j0, j1, 0.0
    weight = (target - v0) / (v1 - v0)
    return j0, j1, min(max(weight, 0.0), 1.0)


def _bilinear(values, xweight, yweight):
    if values.ndim == 2:
        q11 = values[0, 0]
        q21 = values[0, 1]
        q12 = values[1, 0]
        q22 = values[1, 1]
    else:
        q11 = values[:, 0, 0]
        q21 = values[:, 0, 1]
        q12 = values[:, 1, 0]
        q22 = values[:, 1, 1]

    return (
        q11 * (1.0 - xweight) * (1.0 - yweight)
        + q21 * xweight * (1.0 - yweight)
        + q12 * (1.0 - xweight) * yweight
        + q22 * xweight * yweight
    )
