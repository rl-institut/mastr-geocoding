from __future__ import annotations

import zipfile

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from loguru import logger

from mastr_geocoding.config.config import settings

MASTR_DATA_DIR = (
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


def isfloat(num: str):
    """
    Determine if string can be converted to float.

    Parameters
    -----------
    num : str
        String to parse.
    Returns
    -------
    bool
        Returns True in string can be parsed to float.
    """
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
    dump_date = mastr_data.dump_date
    zip_name = mastr_data.zip_name.format(dump_date).split(".")[0]
    data_dir = MASTR_DATA_DIR / zip_name

    res_lst = []

    logger.info(f"Reading MaStR data from {data_dir} ...")

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
            data_dir / file,
            usecols=cols,
            low_memory=False,
        )

        logger.debug(f"Read {data_dir / file}.")

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
            f"{len(ok_df)} of {len(df)} values within {file} have correct values for "
            f"ZIP code and municipality."
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
    Download and extract MaStR data from Zenodo if not already present.
    """
    MASTR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Preparing to download MaStR data to {MASTR_DATA_DIR} ...")

    mastr_data = settings["mastr-data"]
    zenodo_files_url = mastr_data.url.format(mastr_data.deposit_id)
    dump_date = mastr_data.dump_date
    zip_name = mastr_data.zip_name.format(dump_date)
    zip_path = MASTR_DATA_DIR / zip_name

    # Download ZIP if not already present
    if zip_path.exists():
        logger.info(f"ZIP file already exists at {zip_path}, skipping download.")
    else:
        logger.info(f"Downloading ZIP from {zenodo_files_url + zip_name} ...")
        urlretrieve(zenodo_files_url + zip_name, zip_path)
        logger.info(f"Download complete: {zip_path}")

    # Check if all extracted files already exist
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        missing_files = []

        for member in members:
            # Überspringe Verzeichnisse
            if member.endswith("/"):
                continue
            target_path = MASTR_DATA_DIR / member
            if not target_path.exists():
                missing_files.append(member)

        if not missing_files:
            logger.info("All ZIP contents already extracted, skipping extraction.")
        else:
            logger.info(
                f"{len(missing_files)} files missing, extracting ZIP to "
                f"{MASTR_DATA_DIR} ..."
            )
            zip_ref.extractall(MASTR_DATA_DIR)
            logger.info("Extraction complete.")
