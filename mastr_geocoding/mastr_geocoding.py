from __future__ import annotations

import geopandas as gpd
import pandas as pd

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from loguru import logger

from mastr_geocoding.config.config import settings
from mastr_geocoding.data_io import (
    WORKING_DIR_MASTR,
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
    logger.info(f"Geocoding {len(geocoding_df)} locations ...")

    geocode_df = geocoding_df.assign(
        location=geocoding_df.zip_and_municipality.apply(ratelimiter)
    )

    logger.info("Geocoding done.")

    geocode_df = geocode_df.assign(
        point=geocode_df.location.apply(lambda loc: tuple(loc.point) if loc else None)
    )

    geocode_df[["latitude", "longitude", "altitude"]] = pd.DataFrame(
        geocode_df.point.tolist(), index=geocode_df.index
    )

    return gpd.GeoDataFrame(
        geocode_df,
        geometry=gpd.points_from_xy(geocode_df.longitude, geocode_df.latitude),
        crs=f"EPSG:{epsg}",
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
        WORKING_DIR_MASTR.parent
        / settings["geocoding"].export_f.format(
            WORKING_DIR_MASTR.parts[-1], settings["mastr-data"].deposit_id
        ),
        driver="GPKG",
    )
