from __future__ import annotations

import geopandas as gpd
import pandas as pd

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from loguru import logger

from mastr_geocoding.config.config import settings
from mastr_geocoding.data_io import (
    MASTR_DATA_DIR,
    download_mastr_data,
    get_zip_and_municipality,
)


def geocoder(
    user_agent: str,
    min_delay_seconds: int,
) -> RateLimiter:
    """
    Setup Nominatim geocoding class.
    Parameters
    -----------
    user_agent : str
        The app name.
    min_delay_seconds : int
        Delay in seconds to use between requests to Nominatim.
        A minimum of 1 is advised.
    Returns
    -------
    geopy.extra.rate_limiter.RateLimiter
        Nominatim RateLimiter geocoding class to use for geocoding.
    """
    logger.info(f"Setting up rate limiter {user_agent} ...")

    locator = Nominatim(user_agent=user_agent)

    return RateLimiter(
        locator.geocode,
        min_delay_seconds=min_delay_seconds,
    )


def geocode_data(
    geocoding_df: pd.DataFrame,
    ratelimiter: RateLimiter,
    epsg: int,
) -> gpd.GeoDataFrame:
    """
    Geocode zip code and municipality.
    Extract latitude, longitude and altitude.
    Transfrom latitude and longitude to shapely
    Point and return a geopandas GeoDataFrame.
    Parameters
    -----------
    geocoding_df : pandas.DataFrame
        DataFrame containing all unique combinations of
        zip codes with municipalities for geocoding.
    ratelimiter : geopy.extra.rate_limiter.RateLimiter
        Nominatim RateLimiter geocoding class to use for geocoding.
    epsg : int
        EPSG ID to use as CRS.
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing all unique combinations of
        zip codes with municipalities with matching geolocation.
    """
    init_len = len(geocoding_df)
    logger.info(f"Geocoding {init_len} locations ...")

    geocode_df = geocoding_df.assign(
        location=geocoding_df.zip_and_municipality.apply(ratelimiter)
    )

    geocode_df = geocode_df.assign(
        point=geocode_df.location.apply(lambda loc: tuple(loc.point) if loc else None)
    )

    geocode_df[["latitude", "longitude", "altitude"]] = pd.DataFrame(
        geocode_df.point.tolist(), index=geocode_df.index
    )

    failed = geocode_df.loc[geocode_df.latitude.isna() | geocode_df.longitude.isna()]
    len_failed = len(failed)

    logger.info(
        f"Geocoding done. {len_failed} locations of {init_len} could not be parsed. "
        f"Failed: {failed.zip_and_municipality.tolist()}"
    )

    return gpd.GeoDataFrame(
        geocode_df,
        geometry=gpd.points_from_xy(geocode_df.longitude, geocode_df.latitude),
        crs=f"EPSG:{epsg}",  # noqa: E231
    )


def run_mastr_geocoding() -> None:
    """
    Main run function.
    """
    download_mastr_data()

    geocoding_df = get_zip_and_municipality()

    ratelimiter = geocoder(
        settings["geocoding"].user_agent, settings["geocoding"].min_delay_seconds
    )

    geocoded_gdf = geocode_data(
        geocoding_df, ratelimiter, epsg=settings["mastr-data"].epsg
    )

    geocoded_gdf.drop(columns=["location", "point"]).to_file(
        MASTR_DATA_DIR.parent
        / settings["geocoding"].export_f.format(
            MASTR_DATA_DIR.parts[-1], settings["mastr-data"].deposit_id
        ),
        driver="GPKG",
    )
