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

RESULTS_DIR = MASTR_DATA_DIR.parent / f"geocoded_results_{MASTR_DATA_DIR.parts[-1]}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def geocoder(
    user_agent: str,
    min_delay_seconds: int,
    max_retries: int,
    error_wait_seconds: int,
) -> RateLimiter:
    """
    Set up a Nominatim geocoding RateLimiter instance.

    Parameters
    ----------
    user_agent : str
        Name of the application, required by Nominatim for identification.
    min_delay_seconds : int
        Minimum delay (in seconds) between consecutive requests to avoid rate limits.
        A value of at least 1 second is recommended.
    max_retries : int
        Maximum number of retries in case of request failures (e.g., timeouts).
    error_wait_seconds : int
        Waiting time (in seconds) before retrying after an error.

    Returns
    -------
    geopy.extra.rate_limiter.RateLimiter
        Configured RateLimiter for Nominatim geocoding requests.
    """
    logger.info(f"Setting up rate limiter {user_agent} ...")

    locator = Nominatim(user_agent=user_agent)

    return RateLimiter(
        locator.geocode,
        min_delay_seconds=min_delay_seconds,
        max_retries=max_retries,
        error_wait_seconds=error_wait_seconds,
        swallow_exceptions=False,
    )


def try_geocode(text: str, ratelimiter: RateLimiter) -> tuple:
    """
    Try to geocode using full 'PLZ Ort, Deutschland' first,
    then fallback to just 'Ort, Deutschland' if needed.

    Returns
    -------
    (geopy Location or None, str: 'original', 'fallback' or 'failed')
    """
    try:
        result = ratelimiter(text)
        if result:
            return result, "original"

        if "," in text:
            # fallback: drop PLZ and try just city
            fallback = text.split(",")[0].split(" ", 1)[-1] + ", Deutschland"
            result = ratelimiter(fallback)
            if result:
                return result, "fallback"
    except Exception as e:
        logger.warning(f"Error geocoding '{text}': {e}")

    return None, "failed"


def safe_geocode(text: str, ratelimiter: RateLimiter) -> tuple:
    """
    Wrapper around `try_geocode` to catch and log unexpected exceptions.

    Parameters
    ----------
    text : str
        Input text to be geocoded, typically in the format 'PLZ Ort, Deutschland'.
    ratelimiter : geopy.extra.rate_limiter.RateLimiter
        Configured RateLimiter instance used to throttle geocoding requests.

    Returns
    -------
    tuple
        A tuple of (Location or None, str), where the second element indicates the
        source: 'original', 'fallback', 'failed', or 'exception' if an error occurred
        outside the intended fallback logic.
    """
    try:
        return try_geocode(text, ratelimiter)
    except Exception as e:
        logger.warning(f"Exception during geocoding '{text}': {e}")
        return None, "exception"


def geocode_data(
    geocoding_df: pd.DataFrame,
    ratelimiter: RateLimiter,
    epsg: int,
) -> gpd.GeoDataFrame:
    """
    Geocode zip code and municipality.
    Extract latitude, longitude and altitude.
    Transform latitude and longitude to shapely
    Point and return a geopandas GeoDataFrame.
    """
    cache_path = RESULTS_DIR / "geocode_results.csv"

    if cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        cached_successful = cached_df.dropna(subset=["latitude", "longitude"])
        logger.info(
            f"Loaded {len(cached_successful)} successful cached results from "
            f"{cache_path}"
        )
    else:
        cached_successful = pd.DataFrame(columns=["zip_and_municipality"])

    to_geocode_df = geocoding_df[
        ~geocoding_df.zip_and_municipality.isin(cached_successful.zip_and_municipality)
    ].copy()

    if to_geocode_df.empty:
        logger.info("No new locations to geocode. Using cached results only.")
        final_df = cached_successful
    else:
        logger.info(
            f"Geocoding {len(to_geocode_df)} of {len(geocoding_df)} locations ..."
        )

        new_df = pd.DataFrame()

        new_df[
            ["location", "geocode_source"]
        ] = to_geocode_df.zip_and_municipality.apply(
            lambda x: pd.Series(safe_geocode(x, ratelimiter))
        )

        new_df = new_df.assign(
            point=new_df.location.apply(lambda loc: tuple(loc.point) if loc else None)
        )

        new_df[["latitude", "longitude", "altitude"]] = pd.DataFrame(
            new_df.point.tolist(), index=new_df.index
        )

        new_df["zip_and_municipality"] = to_geocode_df.zip_and_municipality.values

        final_df = pd.concat([cached_successful, new_df], ignore_index=True)

        # Speichere vollständige aktualisierte Ergebnisse
        final_df.to_csv(cache_path, index=False)
        logger.info(f"Updated geocode results saved to {cache_path}")

        # Speichere fehlgeschlagene
        failed = new_df.loc[new_df.latitude.isna() | new_df.longitude.isna()]
        len_failed = len(failed)

        if len_failed > 0:
            failed_path = RESULTS_DIR / "failed_geocodes.csv"
            failed[["zip_and_municipality", "geocode_source"]].to_csv(
                failed_path, index=False
            )
            logger.warning(
                f"{len_failed} locations could not be geocoded. Saved to {failed_path}."
            )
        else:
            logger.info("All new locations were successfully geocoded.")

        logger.info(
            f"Geocoding complete. Sources used: "
            f"{new_df.geocode_source.value_counts().to_dict()}"
        )

    return gpd.GeoDataFrame(
        final_df,
        geometry=gpd.points_from_xy(final_df.longitude, final_df.latitude),
        crs=f"EPSG:{epsg}",  # noqa: E231
    )


def run_mastr_geocoding() -> None:
    """
    Main run function.
    """
    download_mastr_data()

    geocoding_df = get_zip_and_municipality()

    ratelimiter = geocoder(
        settings["geocoding"].user_agent,
        settings["geocoding"].min_delay_seconds,
        settings["geocoding"].max_retries,
        settings["geocoding"].error_wait_seconds,
    )

    geocoded_gdf = geocode_data(
        geocoding_df, ratelimiter, epsg=settings["mastr-data"].epsg
    )

    geocoded_gdf.drop(columns=["location", "point"]).to_file(
        RESULTS_DIR
        / settings["geocoding"].export_f.format(
            MASTR_DATA_DIR.parts[-1], settings["mastr-data"].deposit_id
        ),
        driver="GPKG",
    )
