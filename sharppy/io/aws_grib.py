import os
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

import numpy as np

import sharppy.sharptab.profile as profile
import sharppy.sharptab.prof_collection as prof_collection
import sharppy.sharptab.thermo as thermo
import sharppy.sharptab.utils as utils

try:
    import cfgrib
except ImportError:
    cfgrib = None


AWS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".sharppy", "cache", "aws_grib")

MODEL_SPECS = {
    "GFS": {
        "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date:%Y%m%d}/{date:%H}/atmos/gfs.t{date:%H}z.pgrb2.0p25.f{fhour:03d}",
        "interp": "bilinear",
    },
    "HRRR": {
        "url": "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date:%Y%m%d}/conus/hrrr.t{date:%H}z.wrfprsf{fhour:02d}.grib2",
        "interp": "idw",
    },
}

ISO_MIN_PRESSURE = 100
ISO_MAX_PRESSURE = 1000
ISO_VARS = ("HGT", "TMP", "RH", "UGRD", "VGRD")
SURFACE_VARS = (
    ("PRES", "surface"),
    ("HGT", "surface"),
    ("TMP", "2 m above ground"),
    ("DPT", "2 m above ground"),
    ("UGRD", "10 m above ground"),
    ("VGRD", "10 m above ground"),
)


def is_point_forecast_model(model_name):
    return model_name in MODEL_SPECS


def build_profile_collection(model_name, point, run_dt, forecast_hours):
    if cfgrib is None:
        raise ImportError("Point-forecast support requires the 'cfgrib' package.")

    fetcher = AwsGribPointFetcher(model_name, run_dt)
    profiles = []
    dates = []

    for fhour in forecast_hours:
        ds_map = fetcher.open_forecast_hour(fhour)
        profile_obj, valid_dt = fetcher.create_profile(point, ds_map)
        profiles.append(profile_obj)
        dates.append(valid_dt)

    collection = prof_collection.ProfCollection({"": profiles}, dates)
    collection.setMeta("loc", _point_label(point))
    collection.setMeta("observed", False)
    collection.setMeta("base_time", run_dt)
    return collection


class AwsGribPointFetcher(object):
    def __init__(self, model_name, run_dt):
        if model_name not in MODEL_SPECS:
            raise ValueError("Unsupported point-forecast model '%s'." % model_name)

        self.model_name = model_name
        self.spec = MODEL_SPECS[model_name]
        self.run_dt = run_dt

        cache_dir = os.path.join(AWS_CACHE_DIR, model_name.lower(), run_dt.strftime("%Y%m%d%H"))
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def open_forecast_hour(self, fhour):
        grib_path = self._ensure_subset_file(fhour)
        datasets = cfgrib.open_datasets(grib_path, indexpath="")

        ds_map = {}
        for ds in datasets:
            var_names = set(ds.data_vars)
            if set(["t", "u", "v", "gh"]).issubset(var_names):
                ds_map["isobaric"] = ds
            elif set(["sp", "orog"]).issubset(var_names):
                ds_map["surface"] = ds
            elif set(["t2m", "d2m"]).issubset(var_names):
                ds_map["2m"] = ds
            elif set(["u10", "v10"]).issubset(var_names):
                ds_map["10m"] = ds

        missing = [key for key in ("isobaric", "surface", "2m", "10m") if key not in ds_map]
        if missing:
            raise IOError(
                "Decoded AWS GRIB subset for %s is missing sections: %s"
                % (self.model_name, ", ".join(missing))
            )
        return ds_map

    def create_profile(self, point, ds_map):
        lat = float(point["lat"])
        lon = float(point["lon"])

        if self.spec["interp"] == "bilinear":
            sample = _sample_regular_grid(ds_map, lat, lon)
        else:
            sample = _sample_curvilinear_grid(ds_map, lat, lon)

        valid_dt = _to_datetime(ds_map["surface"].coords["valid_time"].values)
        prof = _build_profile_from_sample(point, valid_dt, sample)
        return prof, valid_dt

    def _ensure_subset_file(self, fhour):
        path = os.path.join(self.cache_dir, "f%03d.grib2" % fhour)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path

        url = self.spec["url"].format(date=self.run_dt, fhour=fhour)
        records = self._select_records(url)

        tmp_path = path + ".part"
        with open(tmp_path, "wb") as out:
            for start, end in records:
                range_header = "bytes=%d-%d" % (start, end - 1) if end is not None else "bytes=%d-" % start
                req = Request(url, headers={"Range": range_header})
                out.write(urlopen(req).read())

        os.replace(tmp_path, path)
        return path

    def _select_records(self, url):
        idx_lines = urlopen(url + ".idx").read().decode("utf-8").splitlines()
        idx = [_parse_idx_line(line) for line in idx_lines if line.strip()]

        pressure_levels = []
        available = set((rec["var"], rec["level"]) for rec in idx)
        for rec in idx:
            if rec["var"] != "HGT" or not rec["level"].endswith(" mb"):
                continue

            level_value = int(float(rec["level"].split()[0]))
            if level_value < ISO_MIN_PRESSURE or level_value > ISO_MAX_PRESSURE:
                continue

            if all((var, "%d mb" % level_value) in available for var in ISO_VARS):
                pressure_levels.append(level_value)

        pressure_levels = sorted(set(pressure_levels), reverse=True)
        if len(pressure_levels) == 0:
            raise IOError("No usable isobaric pressure levels found in %s." % url)

        ranges = []
        for level in pressure_levels:
            pairs = [(var, "%d mb" % level) for var in ISO_VARS]
            ranges.append(_range_for_pairs(idx, pairs))

        ranges.append(_range_for_pairs(idx, [("PRES", "surface"), ("HGT", "surface")]))
        ranges.append(_range_for_pairs(idx, [("TMP", "2 m above ground"), ("DPT", "2 m above ground")]))
        ranges.append(_range_for_pairs(idx, [("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")]))

        ranges.sort()
        return _merge_ranges(ranges)


def _build_profile_from_sample(point, valid_dt, sample):
    surface_pres = float(sample["sp"]) * 0.01
    surface_hght = float(sample["orog"])
    surface_tmpc = thermo.ktoc(float(sample["t2m"]))
    surface_dwpc = thermo.ktoc(float(sample["d2m"]))
    surface_u = utils.MS2KTS(float(sample["u10"]))
    surface_v = utils.MS2KTS(float(sample["v10"]))

    pres = np.asarray(sample["levels"], dtype=float)
    hght = np.asarray(sample["gh"], dtype=float)
    tmpc = thermo.ktoc(np.asarray(sample["t"], dtype=float))
    u = utils.MS2KTS(np.asarray(sample["u"], dtype=float))
    v = utils.MS2KTS(np.asarray(sample["v"], dtype=float))

    if "dpt" in sample:
        dwpc = thermo.ktoc(np.asarray(sample["dpt"], dtype=float))
    else:
        rh = np.clip(np.asarray(sample["r"], dtype=float), 0.1, 100.0)
        vapr = np.clip((rh / 100.0) * thermo.vappres(tmpc), 1e-6, None)
        dwpc = thermo.temp_at_vappres(vapr)

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
    pres = pres[order]
    hght = hght[order]
    tmpc = tmpc[order]
    dwpc = dwpc[order]
    u = u[order]
    v = v[order]

    prof = profile.create_profile(
        profile="raw",
        pres=pres,
        hght=hght,
        tmpc=tmpc,
        dwpc=dwpc,
        u=u,
        v=v,
        location=_point_label(point),
        date=valid_dt,
        latitude=float(point["lat"]),
        missing=-9999.0,
        strictQC=False,
    )
    return prof


def _sample_regular_grid(ds_map, lat, lon):
    iso = ds_map["isobaric"]
    surface = ds_map["surface"]
    two_m = ds_map["2m"]
    ten_m = ds_map["10m"]

    lat_vals = np.asarray(iso.latitude.values, dtype=float)
    lon_vals = np.asarray(iso.longitude.values, dtype=float)
    lon = lon % 360.0

    i0, i1, yweight = _bounding_indices(lat_vals, lat)
    j0, j1, xweight = _bounding_indices_cyclic(lon_vals, lon)

    def interp(data):
        values = np.asarray(data.values, dtype=float)
        return _bilinear(values, i0, i1, j0, j1, xweight, yweight)

    return {
        "levels": np.asarray(iso.isobaricInhPa.values, dtype=float),
        "t": interp(iso["t"]),
        "u": interp(iso["u"]),
        "v": interp(iso["v"]),
        "gh": interp(iso["gh"]),
        "r": interp(iso["r"]),
        "sp": float(interp(surface["sp"])),
        "orog": float(interp(surface["orog"])),
        "t2m": float(interp(two_m["t2m"])),
        "d2m": float(interp(two_m["d2m"])),
        "u10": float(interp(ten_m["u10"])),
        "v10": float(interp(ten_m["v10"])),
    }


def _sample_curvilinear_grid(ds_map, lat, lon):
    iso = ds_map["isobaric"]
    surface = ds_map["surface"]
    two_m = ds_map["2m"]
    ten_m = ds_map["10m"]

    lats = np.asarray(iso.latitude.values, dtype=float)
    lons = _wrap180(np.asarray(iso.longitude.values, dtype=float))
    lon = _wrap180(lon)

    coslat = np.cos(np.radians(lat))
    dist2 = ((lats - lat) ** 2) + (((lons - lon) * coslat) ** 2)

    nearest = np.argpartition(dist2.ravel(), 4)[:4]
    iy, ix = np.unravel_index(nearest, dist2.shape)
    weights = 1.0 / np.maximum(dist2[iy, ix], 1e-12)
    weights /= weights.sum()

    def interp(data):
        values = np.asarray(data.values, dtype=float)
        if values.ndim == 3:
            return np.tensordot(values[:, iy, ix], weights, axes=(1, 0))
        return np.dot(values[iy, ix], weights)

    sample = {
        "levels": np.asarray(iso.isobaricInhPa.values, dtype=float),
        "t": interp(iso["t"]),
        "u": interp(iso["u"]),
        "v": interp(iso["v"]),
        "gh": interp(iso["gh"]),
        "r": interp(iso["r"]),
        "sp": float(interp(surface["sp"])),
        "orog": float(interp(surface["orog"])),
        "t2m": float(interp(two_m["t2m"])),
        "d2m": float(interp(two_m["d2m"])),
        "u10": float(interp(ten_m["u10"])),
        "v10": float(interp(ten_m["v10"])),
    }
    if "dpt" in iso:
        sample["dpt"] = interp(iso["dpt"])
    return sample


def _point_label(point):
    return "%.3f, %.3f" % (float(point["lat"]), float(point["lon"]))


def _parse_idx_line(line):
    parts = line.split(":")
    return {"offset": int(parts[1]), "var": parts[3], "level": parts[4]}


def _find_offset(records, var, level):
    for rec in records:
        if rec["var"] == var and rec["level"] == level:
            return rec["offset"]
    raise IOError("Could not find %s at %s in GRIB index." % (var, level))


def _next_offset(records, var, level):
    for idx, rec in enumerate(records):
        if rec["var"] == var and rec["level"] == level:
            if idx + 1 < len(records):
                return records[idx + 1]["offset"]
            return None
    raise IOError("Could not find %s at %s in GRIB index." % (var, level))


def _range_for_pairs(records, pairs):
    starts = [_find_offset(records, var, level) for var, level in pairs]
    last_start = max(starts)
    last_pair = pairs[starts.index(last_start)]
    return min(starts), _next_offset(records, last_pair[0], last_pair[1])


def _merge_ranges(ranges):
    merged = []
    for start, end in ranges:
        if not merged:
            merged.append([start, end])
            continue

        prev_start, prev_end = merged[-1]
        if prev_end is None or start <= prev_end:
            if prev_end is None or end is None:
                merged[-1][1] = None
            else:
                merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])

    return [tuple(item) for item in merged]


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


def _bilinear(values, i0, i1, j0, j1, xweight, yweight):
    if values.ndim == 2:
        q11 = values[i0, j0]
        q21 = values[i0, j1]
        q12 = values[i1, j0]
        q22 = values[i1, j1]
    else:
        q11 = values[:, i0, j0]
        q21 = values[:, i0, j1]
        q12 = values[:, i1, j0]
        q22 = values[:, i1, j1]

    return (
        q11 * (1.0 - xweight) * (1.0 - yweight)
        + q21 * xweight * (1.0 - yweight)
        + q12 * (1.0 - xweight) * yweight
        + q22 * xweight * yweight
    )


def _wrap180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def _to_datetime(value):
    epoch = np.datetime64("1970-01-01T00:00:00")
    seconds = int((value - epoch) / np.timedelta64(1, "s"))
    return datetime.utcfromtimestamp(seconds)
