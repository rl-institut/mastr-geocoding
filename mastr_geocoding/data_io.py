from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from loguru import logger

from mastr_geocoding.config.config import settings

WORKING_DIR_MASTR = (
    Path(__file__).resolve().parent.parent
    / "bnetza_mastr"
    / f"dump_{settings['mastr-data'].dump_date}"
)


def geocoding_data(
    data: np.ndarray,
) -> pd.DataFrame:
    """
    Setup DataFrame to geocode.

    Parameters
    -----------
    data : numpy.ndarray
        numpy.ndarray containing all unique combinations of ZIP code and municipality.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing all unique combinations of
        zip codes with municipalities for geocoding.
    """
    return pd.DataFrame(
        data=data,
        columns=["zip_and_municipality"],
    )


def get_zip_and_municipality() -> pd.DataFrame:
    """
    Setup DataFrame to geocode.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all unique combinations of
        zip codes with municipalities for geocoding.
    """
    mastr_data = settings["mastr-data"]
    f_name = mastr_data.f_name
    technologies = mastr_data.technologies
    federal_state = mastr_data.federal_state

    res_lst = []

    logger.info(f"Reading MaStR data from {WORKING_DIR_MASTR} ...")

    federal_states = []

    for tech in technologies:
        df = pd.read_csv(
            WORKING_DIR_MASTR / f_name.format(tech),
            usecols=["Postleitzahl", "Gemeinde", "Bundesland", "Land"],
            low_memory=False,
        )

        logger.debug(f"Read {WORKING_DIR_MASTR / f_name.format(tech)}.")

        federal_states.extend(df.Bundesland.unique().tolist())

        federal_states = list(set(federal_states))

        if federal_state in federal_states:
            logger.debug(f"Only using data for federal state {federal_state}.")
            df = df.loc[df.Bundesland == federal_state]

        # cleaning plz
        df = df[df["Postleitzahl"].apply(lambda x: str(x).isdigit())]
        df = df.dropna(subset="Postleitzahl")

        res_lst.append(
            df.Postleitzahl.astype(int).astype(str).str.zfill(5)
            + " "
            + df.Gemeinde.astype(str).str.rstrip().str.lstrip()
            + ", Deutschland"
        )

    return geocoding_data(pd.concat(res_lst, ignore_index=True).unique())


def download_mastr_data():
    """
    Download MaStR data from Zenodo.
    """
    WORKING_DIR_MASTR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading MaStR data to {WORKING_DIR_MASTR} ...")

    # Get parameters from config and set download URL
    mastr_data = settings["mastr-data"]
    zenodo_files_url = mastr_data.url.format(mastr_data.deposit_id)
    f_name = mastr_data.f_name
    technologies = mastr_data.technologies

    files = [f_name.format(technology) for technology in technologies]
    files.append(mastr_data.f_name_2)

    # Retrieve specified files
    for filename in files:
        path = WORKING_DIR_MASTR / filename

        if not path.is_file():
            urlretrieve(zenodo_files_url + filename, path)

            logger.debug(f"Downloaded {filename} from {zenodo_files_url + filename}.")

        else:
            logger.debug(
                f"Already downloaded {filename} from {zenodo_files_url + filename}."
            )
