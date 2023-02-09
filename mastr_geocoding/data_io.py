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


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def zip_and_municipality_from_standort(
    standort: str,
) -> tuple[str, bool]:
    """
    Get zip code and municipality from Standort string split into a list.

    Parameters
    -----------
    standort : str
        Standort as given from MaStR data.
    Returns
    -------
    str
        Standort with only the zip code and municipality
        as well a ', Germany' added.
    """
    standort_list = standort.split()

    found = False
    count = 0

    for count, elem in enumerate(standort_list):
        if len(elem) != 5:
            continue
        if not elem.isnumeric():
            continue

        found = True

        break

    if found:
        cleaned_str = " ".join(standort_list[count:])

        return cleaned_str, found

    logger.warning(
        "Couldn't identify zip code. This entry will be dropped."
        f" Original standort: {standort}."
    )

    return standort, found


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

    federal_states = set()

    base_cols = ["Postleitzahl", "Gemeinde", "Bundesland", "Land"]
    extra = ["Standort"]

    for tech in technologies:
        if tech == "solar":
            cols = base_cols + extra
        else:
            cols = base_cols

        file = f_name.format(tech)

        df = pd.read_csv(
            WORKING_DIR_MASTR / file,
            usecols=cols,
            low_memory=False,
        )

        logger.debug(f"Read {WORKING_DIR_MASTR / file}.")

        federal_states = federal_states.union(set(df.Bundesland.unique()))

        if federal_state in federal_states:
            logger.debug(f"Only using data for federal state {federal_state}.")
            df = df.loc[df.Bundesland == federal_state]

        # cleaning plz
        mask = (
            df.Postleitzahl.apply(isfloat)
            & ~df.Postleitzahl.isna()
            & ~df.Gemeinde.isna()
        )
        ok_df = df.loc[mask]

        logger.info(
            f"{len(ok_df)} of {len(df)} within {file} have correct values for ZIP code "
            f"and municipality."
        )

        res_lst.append(
            ok_df.Postleitzahl.astype(int).astype(str).str.zfill(5)
            + " "
            + ok_df.Gemeinde.astype(str).str.rstrip().str.lstrip()
            + ", Deutschland"
        )

        # get zip and municipality from Standort
        parse_df = df.loc[~mask]

        if parse_df.empty or "Standort" not in parse_df.columns:
            continue

        init_len = len(parse_df)

        logger.info(
            f"Parsing ZIP code and municipality from Standort for {init_len} values "
            f"for {file}."
        )

        parsed_df = pd.DataFrame(
            parse_df.Standort.astype(str)
            .apply(zip_and_municipality_from_standort)
            .tolist(),
            index=parse_df.index,
            columns=["zip_and_municipality", "drop_this"],
        )

        parsed_df = parsed_df.loc[parsed_df.drop_this]

        logger.info(
            f"Successfully parsed {len(parsed_df)} of {init_len} values for {file}."
        )

        res_lst.append(parsed_df["zip_and_municipality"] + ", Deutschland")

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
