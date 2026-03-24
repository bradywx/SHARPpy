from sharppy.io.aws_grib import build_profile_collection as build_aws_profile_collection
from sharppy.io.aws_grib import is_point_forecast_model as is_aws_point_forecast_model
from sharppy.io.era5_zarr import build_profile_collection as build_era5_profile_collection


def is_point_forecast_model(model_name):
    return model_name == "ERA5" or is_aws_point_forecast_model(model_name)


def build_profile_collection(model_name, point, run_dt, forecast_hours):
    if model_name == "ERA5":
        return build_era5_profile_collection(point, run_dt, forecast_hours)
    return build_aws_profile_collection(model_name, point, run_dt, forecast_hours)
